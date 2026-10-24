# PQ threshold bounded reservation

`PARASELECT_PQ_THRESHOLD_RESERVATION=scalar|warp` selects candidate reservation
for both plain and residual PQ threshold scans. Default: `scalar`.

Both retain arriving candidates up to the fixed capacity, without sorting or
replacing retained candidates. `warp` combines the currently active accepted
lanes into one bounded CAS reservation. Remaining capacity can admit only part
of a warp. The shared count never exceeds capacity. Lane scheduling and block
arrival can change retained IDs, so compare recall as well as timing.

`PARASELECT_PQ_THRESHOLD_AUDIT=1` additionally counts all accepted candidates,
including those dropped. It validates `kept=min(accepted,capacity)` per query.
Audit is disabled for timing comparisons because its extra atomics add work.

Standalone GPU test (from repository root):

```sh
/usr/local/cuda-13.0/bin/nvcc -O3 -arch=sm_120 \
  -Isrc/search/kernels/fine/gpu_fine/pq_adc \
  tests/smoke/pq_threshold/test_bounded_reservation.cu -o /tmp/test_pq_reservation
CUDA_VISIBLE_DEVICES=6 /tmp/test_pq_reservation
CUDA_VISIBLE_DEVICES=6 /usr/local/cuda-13.0/bin/compute-sanitizer \
  --tool synccheck --error-exitcode 5 /tmp/test_pq_reservation
```

The tests cover 288 combinations of input length, capacity and accepted-lane
spacing, checking counts, guard slots, uniqueness and accepted-only output.
