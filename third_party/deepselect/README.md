# DeepSelect FP32 SM120 port

Vendored from https://github.com/deepseek-ai/DeepSelect at
0f03b68748b304863fdf0181a11458d04ae533a9 (MIT; license retained).
Only FP32 core and kerutils headers are included. CUTLASS headers are an external
build dependency, pinned upstream submodule ae6bccf341fb4410241f696ba06873023d5ce4ed.

Local changes: allow 128 threads per CTA and a max-k=2048 instantiation.
Wrapper uses 128 threads, 2048 elements/round, reconstruct threshold=2048,
two TMA stages. These configurations are experimental until correctness tests pass.
The upstream FP32 radix/filter algorithm and value-sorted output are retained.

Tail length is limited to the round size. The index-sorting buffer assertion is
conditional on sorted_index; this port only instantiates value sorting.
