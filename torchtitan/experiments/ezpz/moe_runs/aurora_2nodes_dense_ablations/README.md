# Aurora 2-Node Dense Ablations

These configs use the dense active-matched DeepSeek-V3 shape:

- 27 dense layers
- FFN hidden dim 7040
- sequence length 4096
- flex attention
- `torch.compile` on model blocks and loss

The baseline comparison was:

- `data_parallel_replicate_degree=2`
- `data_parallel_shard_degree=12`
- `local_batch_size=2`
- selective op activation checkpointing
- default FSDP resharding

The ablations isolate these knobs:

- `deepseek_v3_dense_active_matched_ddp24_lb2_ac_flex_compile.json`: removes FSDP sharding by using 24-way replicated DP.
- `deepseek_v3_dense_active_matched_hsdp12_no_reshard_lb2_ac_flex_compile.json`: keeps HSDP but disables FSDP reshard-after-forward.
- `deepseek_v3_dense_active_matched_hsdp12_no_ac_lb2_flex_compile.json`: keeps baseline HSDP but disables activation checkpointing.
- `deepseek_v3_dense_active_matched_hsdp12_lb4_ac_flex_compile.json`: keeps baseline HSDP and AC but doubles local batch size.
