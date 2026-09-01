"""Phase-separated shared-expert work for native EP communication windows."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from aurora_moe._kernels.shared_expert_bmm import shared_expert_loop


def _record(name: str):
    if os.environ.get("AURORA_MOE_PROFILE") == "1":
        return torch.profiler.record_function(name)
    return nullcontext()


class PhaseSharedExpertController:
    """Schedule exact shared experts between routed XMX phases on one XPU."""

    def __init__(
        self,
        x: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        needs: tuple[bool, bool, bool, bool],
        stream: torch.xpu.Stream,
    ) -> None:
        if x.device.type != "xpu" or any(tensor.device != x.device for tensor in (up, gate, down)):
            raise ValueError("shared phase controller requires same-device XPU tensors")
        if up.shape != gate.shape or up.ndim != 3 or down.ndim != 3:
            raise ValueError("shared parameters must be up/gate=[S,D,H], down=[S,H,D]")
        if (
            up.size(0) != down.size(0)
            or up.size(1) != x.shape[-1]
            or up.size(2) != down.size(1)
            or down.size(2) != x.shape[-1]
        ):
            raise ValueError("incompatible shared-expert dimensions")
        if not isinstance(stream, torch.xpu.Stream) or stream.device != x.device:
            raise ValueError("shared stream must be an XPU stream on x.device")
        if len(needs) != 4:
            raise ValueError("needs must contain x, up, gate, and down flags")
        self.x = x
        self.up = up
        self.gate = gate
        self.down = down
        self.needs = tuple(bool(value) for value in needs)
        self.stream = stream
        self._split = (up.size(0) + 1) // 2
        self._backward_ranges, self._forward_prefix_chunks = self._phase_chunk_ranges(
            up.size(0), self.backward_chunk_count_requested(), self._split
        )
        self._chunked_backward = len(self._backward_ranges) > 1
        self._dispatch_ready = None
        self._prefix_done = None
        self._return_ready = None
        self._forward_done = None
        self._grad_ready = None
        self._payload_ready = None
        self._early_backward_done = None
        self._backward_done = None
        self._backward_started_early = False
        self._inner = None
        self._prefix = None
        self._shared = None
        self._grad_output = None
        self._chunk_outputs: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._next_backward_chunk = 0
        self._result_accum: list[torch.Tensor | None] = [None] * 4
        self._result: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ] | None = None

    @property
    def shared_experts(self) -> int:
        return self.up.size(0)

    @staticmethod
    def backward_chunk_count_requested() -> int:
        """Return the runtime number of exact shared-expert backward chunks."""

        value = os.environ.get("AURORA_MOE_PHASE_SHARED_BACKWARD_CHUNKS", "1")
        try:
            chunks = int(value)
        except ValueError as error:
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_BACKWARD_CHUNKS must be a positive integer"
            ) from error
        if chunks < 1:
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_BACKWARD_CHUNKS must be a positive integer"
            )
        return chunks

    @staticmethod
    def early_backward_chunk_count_requested() -> int:
        """Return the runtime number of shared-backward chunks placed in A2."""

        value = os.environ.get(
            "AURORA_MOE_PHASE_SHARED_BACKWARD_EARLY_CHUNKS", "1"
        )
        try:
            chunks = int(value)
        except ValueError as error:
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_BACKWARD_EARLY_CHUNKS must be a positive integer"
            ) from error
        if chunks < 1:
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_BACKWARD_EARLY_CHUNKS must be a positive integer"
            )
        return chunks

    @staticmethod
    def _chunk_ranges(total: int, chunks: int) -> tuple[tuple[int, int], ...]:
        if total < 0 or chunks < 1:
            raise ValueError("shared-expert chunk ranges require total >= 0 and chunks >= 1")
        if total == 0:
            return ()
        chunks = min(total, chunks)
        base, extra = divmod(total, chunks)
        start = 0
        ranges = []
        for index in range(chunks):
            stop = start + base + int(index < extra)
            ranges.append((start, stop))
            start = stop
        return tuple(ranges)

    @classmethod
    def _phase_chunk_ranges(
        cls, total: int, chunks: int, forward_split: int
    ) -> tuple[tuple[tuple[int, int], ...], int]:
        """Partition backward without moving the existing A1/A3 forward split."""

        if not 0 <= forward_split <= total:
            raise ValueError("shared forward split must be inside the expert range")
        if total == 0:
            return (), 0
        chunks = min(total, chunks)
        if chunks == 1:
            return ((0, total),), 1
        prefix_experts = forward_split
        suffix_experts = total - forward_split
        if prefix_experts == 0 or suffix_experts == 0:
            return cls._chunk_ranges(total, chunks), chunks

        min_prefix_chunks = max(1, chunks - suffix_experts)
        max_prefix_chunks = min(prefix_experts, chunks - 1)
        desired_prefix_chunks = (chunks * prefix_experts + total // 2) // total
        prefix_chunks = min(
            max(desired_prefix_chunks, min_prefix_chunks), max_prefix_chunks
        )
        suffix_chunks = chunks - prefix_chunks
        prefix = cls._chunk_ranges(prefix_experts, prefix_chunks)
        suffix = tuple(
            (forward_split + start, forward_split + stop)
            for start, stop in cls._chunk_ranges(suffix_experts, suffix_chunks)
        )
        return prefix + suffix, prefix_chunks

    def _event(self, stream: torch.xpu.Stream) -> torch.xpu.Event:
        event = torch.xpu.Event()
        event.record(stream)
        return event

    def _make_inner(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._inner is None:
            self._inner = tuple(
                tensor.detach().requires_grad_(needed)
                for tensor, needed in zip((self.x, self.up, self.gate, self.down), self.needs)
            )
        return self._inner

    def _chunk(
        self, start: int, stop: int, *, initial: torch.Tensor | None = None
    ) -> torch.Tensor:
        x, up, gate, down = self._make_inner()
        flat = x.reshape(-1, x.shape[-1])
        if start == stop:
            return flat.new_zeros(flat.shape) if initial is None else initial
        return shared_expert_loop(
            flat,
            up[start:stop],
            gate[start:stop],
            down[start:stop],
            initial=initial,
        )

    def _independent_chunk(
        self, start: int, stop: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, up, gate, down = self._make_inner()
        flat = x.reshape(-1, x.shape[-1])
        up_chunk = up[start:stop]
        gate_chunk = gate[start:stop]
        down_chunk = down[start:stop]
        output = shared_expert_loop(flat, up_chunk, gate_chunk, down_chunk)
        return output, up_chunk, gate_chunk, down_chunk, flat

    def _run_independent_chunks(
        self, ranges: tuple[tuple[int, int], ...], *, initial: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not ranges:
            if initial is None:
                raise RuntimeError("independent shared chunks must contain at least one range")
            return initial
        total = initial
        for start, stop in ranges:
            with _record(f"moe.phase_shared.forward.chunk_{start}_{stop}"):
                output, up_chunk, gate_chunk, down_chunk, flat = self._independent_chunk(
                    start, stop
                )
            self._chunk_outputs.append((output, up_chunk, gate_chunk, down_chunk, flat))
            total = output if total is None else total + output
        assert total is not None
        return total

    def _record_shared_stream(self, *tensors: torch.Tensor | None) -> None:
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(self.stream)

    def mark_dispatch_ready(self, stream: torch.xpu.Stream) -> None:
        """Capture the packed-route producer event before native A1 submission."""

        if self._dispatch_ready is not None:
            raise RuntimeError("dispatch readiness may be marked only once")
        self._dispatch_ready = self._event(stream)

    def start_prefix_after_dispatch(self) -> None:
        """Enqueue the first shared chunk after native A1 is submitted."""

        if self._dispatch_ready is None or self._prefix_done is not None:
            raise RuntimeError("mark dispatch readiness before starting the shared prefix")
        with torch.enable_grad(), torch.xpu.stream(self.stream):
            self.stream.wait_event(self._dispatch_ready)
            if self._chunked_backward:
                self._prefix = self._run_independent_chunks(
                    self._backward_ranges[: self._forward_prefix_chunks]
                ).detach()
            else:
                self._prefix = self._chunk(0, self._split)
            self._record_shared_stream(
                self.x, self.up, self.gate, self.down, *self._make_inner(), self._prefix
            )
            self._prefix_done = self._event(self.stream)

    def wait_prefix_before_local_xmx(self, stream: torch.xpu.Stream) -> None:
        """Prevent routed local GEMMs from overlapping the shared prefix."""

        if self._prefix_done is None:
            raise RuntimeError("start the shared prefix before waiting for it")
        stream.wait_event(self._prefix_done)

    def mark_return_ready(self, stream: torch.xpu.Stream) -> None:
        """Capture local-output readiness before native A3 submission."""

        if self._prefix_done is None or self._return_ready is not None:
            raise RuntimeError("start one prefix and mark return readiness only once")
        self._return_ready = self._event(stream)

    def start_suffix_after_return(self) -> None:
        """Enqueue remaining shared experts after native A3 is submitted."""

        if self._return_ready is None or self._forward_done is not None:
            raise RuntimeError("mark return readiness before starting the shared suffix")
        with torch.enable_grad(), torch.xpu.stream(self.stream):
            self.stream.wait_event(self._return_ready)
            if self._chunked_backward:
                if self._prefix is None:
                    raise RuntimeError("chunked shared suffix is missing its forward prefix")
                self._shared = self._run_independent_chunks(
                    self._backward_ranges[self._forward_prefix_chunks :], initial=self._prefix
                ).reshape_as(self._make_inner()[0]).detach()
                self._prefix = None
            else:
                self._shared = self._chunk(
                    self._split, self.shared_experts, initial=self._prefix
                ).reshape_as(self._make_inner()[0])
            self._record_shared_stream(self._shared)
            self._forward_done = self._event(self.stream)

    def finish_forward(self, stream: torch.xpu.Stream) -> torch.Tensor:
        """Join the final shared result only where routed/shared outputs combine."""

        if self._forward_done is None or self._shared is None:
            raise RuntimeError("start the shared suffix before finishing forward")
        stream.wait_event(self._forward_done)
        self._shared.record_stream(stream)
        return self._shared

    def saved_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return nested shared leaves and output for the outer autograd context."""

        if self._inner is None or self._shared is None:
            raise RuntimeError("shared forward has not completed")
        return (*self._inner, self._shared)

    @staticmethod
    def early_backward_requested() -> bool:
        """Whether the exact two-phase routed path may overlap shared dB with A2.

        This is deliberately opt-in because it changes only scheduling: shared
        backward begins after the outer gradient is ready and the routed stream
        later waits for it before its first local XMX operation.  It is valid
        only where the routed implementation installs that reciprocal wait.
        """

        value = os.environ.get("AURORA_MOE_PHASE_SHARED_BACKWARD_EARLY", "0")
        if value not in ("0", "1"):
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_BACKWARD_EARLY must be '0' or '1'"
            )
        return value == "1"

    @staticmethod
    def weight_tail_overlap_requested() -> bool:
        """Whether exact routed dW may share XMX execution with shared dB.

        The normal phase protocol serializes those two compute-heavy regions
        because their aggregate XMX pressure is often worse than a clean
        overlap with communication.  This opt-in is a measured scheduling
        diagnostic only: the tensors are disjoint, but it must pass the full
        distributed multi-step correctness/liveness gate before use.
        """

        value = os.environ.get("AURORA_MOE_PHASE_SHARED_WEIGHT_TAIL_OVERLAP", "0")
        if value not in ("0", "1"):
            raise ValueError(
                "AURORA_MOE_PHASE_SHARED_WEIGHT_TAIL_OVERLAP must be '0' or '1'"
            )
        return value == "1"

    @property
    def backward_started_early(self) -> bool:
        return self._backward_started_early

    @property
    def backward_after_a4_required(self) -> bool:
        """Whether shared backward still has exact chunks pending after A4."""

        return self._backward_done is None

    def arm_backward(self, grad_output: torch.Tensor, stream: torch.xpu.Stream) -> None:
        """Capture outer-gradient readiness before routed reverse communication."""

        if self._shared is None or self._grad_output is not None:
            raise RuntimeError("shared backward may be armed exactly once after forward")
        if stream.device != grad_output.device:
            raise ValueError("shared backward gradient stream must match grad_output")
        self._grad_output = grad_output
        # The consumer stream already waits for the routed forward stream in
        # the joint-autograd wrapper.  Recording here therefore captures both
        # the forward shared state and the outer gradient without a host wait.
        self._grad_ready = self._event(stream)

    def mark_payload_ready(self, stream: torch.xpu.Stream) -> None:
        """Capture routed dX/payload completion before reverse A4 submission."""

        if self._grad_output is None or self._payload_ready is not None:
            raise RuntimeError("arm shared backward before marking payload readiness")
        self._payload_ready = self._event(stream)

    def _start_backward_after(self, ready: torch.xpu.Event, *, early: bool) -> None:
        if self._backward_done is not None:
            raise RuntimeError("shared backward may be started only once")
        assert self._inner is not None and self._shared is not None and self._grad_output is not None
        with torch.enable_grad(), torch.xpu.stream(self.stream):
            self.stream.wait_event(ready)
            self._record_shared_stream(*self._inner, self._shared, self._grad_output)
            if self.shared_experts == 0:
                # The zero-sized loop has no edge to its parameter tensors.
                # Match the ordinary loop's absent parameter gradients, while
                # retaining a zero contribution for x when it needs one.
                self._result = (
                    torch.zeros_like(self._inner[0]) if self.needs[0] else None,
                    None,
                    None,
                    None,
                )
            else:
                pairs = tuple(
                    (index, tensor)
                    for index, (tensor, needed) in enumerate(zip(self._inner, self.needs))
                    if needed
                )
                gradients = torch.autograd.grad(
                    self._shared,
                    tuple(tensor for _, tensor in pairs),
                    self._grad_output,
                    allow_unused=True,
                ) if pairs else ()
                result: list[torch.Tensor | None] = [None] * 4
                for (index, _), gradient in zip(pairs, gradients):
                    result[index] = gradient
                self._result = tuple(result)
            self._record_shared_stream(*self._result)
            self._backward_done = self._event(self.stream)
        self._backward_started_early = self._backward_started_early or early

    def _accumulate_chunk_gradient(
        self,
        index: int,
        gradient: torch.Tensor | None,
        start: int,
        stop: int,
    ) -> None:
        if gradient is None:
            return
        if index == 0:
            previous = self._result_accum[0]
            if previous is None:
                self._result_accum[0] = gradient
            else:
                previous.add_(gradient)
            return
        full = self._result_accum[index]
        if full is None:
            assert self._inner is not None
            full = torch.zeros_like(self._inner[index])
            self._result_accum[index] = full
        full[start:stop].copy_(gradient)

    def _start_chunked_backward_after(
        self, ready: torch.xpu.Event, limit: int, *, early: bool
    ) -> None:
        if not self._chunked_backward:
            raise RuntimeError("chunked shared backward requires multiple live expert ranges")
        if self._backward_done is not None:
            raise RuntimeError("shared backward has already completed")
        if self._inner is None or self._shared is None or self._grad_output is None:
            raise RuntimeError("shared forward and outer gradient must be available before backward")
        if len(self._chunk_outputs) != len(self._backward_ranges):
            raise RuntimeError("shared forward did not retain every requested backward chunk")
        if not self._next_backward_chunk < limit <= len(self._chunk_outputs):
            raise RuntimeError("invalid shared-backward chunk interval")

        with torch.enable_grad(), torch.xpu.stream(self.stream):
            self.stream.wait_event(ready)
            self._record_shared_stream(*self._inner, self._shared, self._grad_output)
            for chunk_index in range(self._next_backward_chunk, limit):
                output, up_chunk, gate_chunk, down_chunk, _flat = self._chunk_outputs[chunk_index]
                start, stop = self._backward_ranges[chunk_index]
                phase = "early" if early else "after_a4"
                with _record(
                    f"moe.phase_shared.backward.{phase}.chunk_{chunk_index}"
                ):
                    pairs: list[tuple[int, torch.Tensor]] = []
                    if self.needs[0]:
                        pairs.append((0, self._inner[0]))
                    if self.needs[1]:
                        pairs.append((1, up_chunk))
                    if self.needs[2]:
                        pairs.append((2, gate_chunk))
                    if self.needs[3]:
                        pairs.append((3, down_chunk))
                    gradients = (
                        torch.autograd.grad(
                            output,
                            tuple(tensor for _, tensor in pairs),
                            self._grad_output.reshape_as(output),
                            allow_unused=True,
                            retain_graph=chunk_index + 1 < len(self._chunk_outputs),
                        )
                        if pairs
                        else ()
                    )
                    for (index, _), gradient in zip(pairs, gradients):
                        self._accumulate_chunk_gradient(index, gradient, start, stop)
                    self._record_shared_stream(
                        output, up_chunk, gate_chunk, down_chunk, *gradients
                    )

            self._next_backward_chunk = limit
            self._record_shared_stream(*self._result_accum)
            if early:
                self._early_backward_done = self._event(self.stream)
            if self._next_backward_chunk == len(self._chunk_outputs):
                self._result = tuple(self._result_accum)
                self._backward_done = self._event(self.stream)
        self._backward_started_early = self._backward_started_early or early

    def start_backward_early(self) -> None:
        """Queue shared dB during routed A2, before its local XMX payload phase.

        The reciprocal ``wait_backward_before_local_xmx`` call is mandatory
        before the routed implementation starts its own payload GEMMs.  Thus
        this method never deliberately overlaps two XMX-heavy backward paths.
        """

        if self._grad_ready is None:
            raise RuntimeError("arm shared backward before starting it early")
        if self._chunked_backward:
            early_chunks = min(
                self.early_backward_chunk_count_requested(), len(self._chunk_outputs)
            )
            self._start_chunked_backward_after(self._grad_ready, early_chunks, early=True)
        else:
            self._start_backward_after(self._grad_ready, early=True)

    def start_backward_after_a4(self) -> None:
        """Run shared backward while reverse A4 is in flight."""

        if self._payload_ready is None or self._backward_done is not None:
            raise RuntimeError("mark payload readiness before shared backward")
        if self._chunked_backward:
            self._start_chunked_backward_after(
                self._payload_ready, len(self._chunk_outputs), early=False
            )
        else:
            self._start_backward_after(self._payload_ready, early=False)

    def wait_backward_before_local_xmx(self, stream: torch.xpu.Stream) -> None:
        """Keep routed local backward GEMMs disjoint from early shared dB."""

        if not self._backward_started_early:
            raise RuntimeError("early shared backward was not started")
        ready = (
            self._early_backward_done
            if self._early_backward_done is not None
            else self._backward_done
        )
        if ready is None:
            raise RuntimeError("early shared backward did not record its completion event")
        stream.wait_event(ready)

    def wait_backward_for_weight_tail(self, stream: torch.xpu.Stream) -> None:
        """Keep routed dW GEMMs disjoint from shared backward GEMMs."""

        if self._backward_done is None:
            raise RuntimeError("start shared backward before gating the weight tail")
        if self.weight_tail_overlap_requested():
            return
        stream.wait_event(self._backward_done)

    def finish_backward(
        self, stream: torch.xpu.Stream
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Join shared gradients after routed backward returns."""

        if self._backward_done is None or self._result is None:
            raise RuntimeError("shared backward has not completed")
        stream.wait_event(self._backward_done)
        for gradient in self._result:
            if gradient is not None:
                gradient.record_stream(stream)
        return self._result
