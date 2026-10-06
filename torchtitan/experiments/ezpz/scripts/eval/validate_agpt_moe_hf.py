import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def validate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    references = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.reference
    ]
    tokens = references[0]["tokens"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_checkpoint, trust_remote_code=True
    )
    seed = tokenizer.encode(
        "Aurora trains sparse language models.", add_special_tokens=True
    )
    if tokens[: len(seed)] != seed:
        raise RuntimeError("Exported tokenizer does not reproduce reference tokens")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    if args.source_dcp:
        import torch.distributed.checkpoint as dcp

        metadata = dcp.FileSystemReader(args.source_dcp).read_metadata()
        biases = {}
        for layer_id, layer in enumerate(model.model.layers):
            prefix = f"layers.{layer_id}.moe."
            key = next(
                (prefix + name for name in ("expert_bias_E", "expert_bias")
                 if prefix + name in metadata.state_dict_metadata), None
            )
            if key is None:
                raise RuntimeError(f"Missing source routing bias in layer {layer_id}")
            value = metadata.state_dict_metadata[key]
            biases[key] = torch.empty(value.size, dtype=value.properties.dtype)
        dcp.load(biases, checkpoint_id=args.source_dcp)
        for layer, bias in zip(model.model.layers, biases.values(), strict=True):
            actual = layer.mlp.expert_bias.cpu()
            if actual.dtype != torch.float32 or not torch.equal(actual, bias):
                raise RuntimeError("Exported routing bias differs from source FP32 bias")
        (output / "routing-bias.json").write_text(json.dumps({
            "layers": len(biases), "dtype": "float32", "exact_match": True,
            "source_dcp": args.source_dcp,
        }, indent=2) + "\n")
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device)[None]
    logits = model(input_ids, use_cache=False).logits.float().cpu()
    comparisons = {}
    for path, reference in zip(args.reference, references, strict=True):
        if reference["tokens"] != tokens:
            raise RuntimeError(f"Reference tokens differ: {path}")
        reference_logits = reference["logits"].float()
        if logits.shape != reference_logits.shape:
            raise RuntimeError(f"Logit shape mismatch for {path}")
        delta = logits - reference_logits
        rms = delta.square().mean().sqrt()
        metrics = {
            "shape": list(logits.shape),
            "max_abs_error": delta.abs().max().item(),
            "mean_abs_error": delta.abs().mean().item(),
            "rms_error": rms.item(),
            "relative_rms_error": (
                rms / reference_logits.square().mean().sqrt()
            ).item(),
            "cosine_similarity": F.cosine_similarity(
                logits.flatten(), reference_logits.flatten(), dim=0
            ).item(),
            "top1_agreement": (
                logits.argmax(-1) == reference_logits.argmax(-1)
            ).float().mean().item(),
            "last_top10_reference": reference_logits[0, -1]
            .topk(10)
            .indices.tolist(),
            "last_top10_candidate": logits[0, -1].topk(10).indices.tolist(),
        }
        comparisons[path] = metrics
        if metrics["relative_rms_error"] > args.max_relative_rms:
            raise RuntimeError(f"Relative RMS error exceeds threshold for {path}")
        if metrics["cosine_similarity"] < args.min_cosine:
            raise RuntimeError(f"Cosine similarity is below threshold for {path}")
        if metrics["top1_agreement"] < args.min_top1_agreement:
            raise RuntimeError(f"Top-1 agreement is below threshold for {path}")
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    torch.save({"tokens": tokens, "logits": logits}, output / "logits.pt")
    print(json.dumps(comparisons, indent=2), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--hf-checkpoint", required=True)
parser.add_argument("--reference", required=True, action="append")
parser.add_argument("--source-dcp", help="Verify every loaded FP32 routing bias exactly")
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="xpu:0")
parser.add_argument("--max-relative-rms", type=float, default=0.02)
parser.add_argument("--min-cosine", type=float, default=0.999)
parser.add_argument("--min-top1-agreement", type=float, default=0.95)
validate(parser.parse_args())
