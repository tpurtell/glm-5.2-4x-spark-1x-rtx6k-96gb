# Balanced-path visual model

This is the measured production-shaped `balanced` path for Luke NVFP4 with the
RedHat dSpark checkpoint. It deliberately collapses the repeated transformer
stack while keeping the ownership, cache, transport, kernel, reduction, and
speculative-control boundaries visible.

[Open the architecture SVG at full size](balanced-path-architecture.svg).

![Balanced-path information flow](balanced-path-architecture.svg)

[Open the timeline SVG at full size](balanced-path-timeline.svg).

![Balanced-path timing](balanced-path-timeline.svg)

## How to read the timing

The green headline numbers are complete uninstrumented requests from commit
`d5a0a367` on the maintained source-tree deployment:

- adaptive dSpark: 27.871 weighted wall TPS over five complete seven-case
  corpus repeats, with repeat results from 27.514 to 28.454 TPS;
- policy behavior: 78.61% proposal acceptance and 2.643 emitted tokens per
  speculative cycle;
- fresh 8K: 8,219 prefill rows in five chunks, 4.787 seconds of server prefill,
  4.793 seconds to the external first token, and 1,715 external prompt TPS;
- all 35 weighted requests added zero runtime graph captures.

Physical target `M` is one target/decode row plus `D` dSpark proposals. The
clean short-context cycle envelope was:

| Physical M | Median cycle | Samples | Meaning |
| ---: | ---: | ---: | --- |
| 1 | about 47.4 ms | 844 | Residual non-verifier decode time per scalar cycle |
| 2 | 72.1 ms | 898 | One target row plus one proposal |
| 3 | 90.8 ms | 152 | One target row plus two proposals |
| 4 | 101.7 ms | 259 | One target row plus three proposals |
| 5 | 114.6 ms | 60 | One target row plus four proposals |
| 6 | 125.1 ms | 144 | One target row plus five proposals |
| 7 | 143.0 ms | 2 | One target row plus six proposals; low sample count |
| 8 | 147.8 ms | 50 | One target row plus seven proposals |

`M=1` is an aggregate residual estimate because the API records direct
verifier-cycle clocks only when dSpark submits proposals. `M=2..8` are direct
medians of 1,565 verifier cycles.

The purple timing breakdown comes from a separate matched deployment with
verbose host-stage clocks and CUDA events enabled. Those clocks synchronize
otherwise asynchronous work: they raised M=3 from 90.8 to 100.8 ms and reduced
the 8K result from 1,715 to 1,405 TPS. They are useful for attribution, not as
production throughput.

| Synchronized M=3 stage | Time |
| --- | ---: |
| Three dense layers: attention | 1.419 ms |
| Three dense layers: numeric work | 2.054 ms |
| 75 sparse layers: attention | 31.529 ms |
| 75 sparse layers: numeric/remote work | 60.253 ms |
| Scheduler, dSpark handoff, and remainder | 5.513 ms |
| Complete synchronized scheduler cycle | 100.768 ms |

Inside the 75-layer sparse section, overlapping host-stage clocks attributed
8.346 ms to shared MLP work, 7.055 ms to routing, and 48.694 ms to Spark
dispatch. They are not additive. The timestamp-isolated Spark trace measured
mean expert-executor time per shard and sparse layer of about 0.241/0.374/0.513
ms at M=1/2/3.

The 8K diagnostic consisted of five rolling chunks: four 1,642-row packs and a
final combined 1,640-row pack including the output row. It made 390 prefill
attention calls and 375 sparse dispatches. The median synchronized
layer/chunk attention stage was 6.258 ms; the median sparse dispatch stage was
40.506 ms. The five chunks overlap across layers, so summing those work clocks
would overstate the 5.841-second diagnostic wall time.

## Residency and data movement

The coordinator owns the residual stream, every attention operation, dense
and shared MLPs, routing, terminal scoring, and both caches. The four Sparks
own only routed-expert weights and execution scratch.

- Target weights resident on the coordinator: 37,202,615,808 bytes.
- Shared target cache: 609,536 physical slots at 56,544 bytes/token, or
  32.1 GiB. Each request is logically capped at 400K tokens.
- Target cache representation: 78 packed FP8 MLA records plus 21 BF16 DSA
  index records. Attention reads it directly; there is no full BF16 unpack.
- dSpark weights: 2,499,130,626 bytes.
- dSpark GPU KV: four independent 2,128-token leases, 408 MiB total, backed by
  three BF16 draft layers with native 2K sliding attention.
- dSpark reusable tails: separate 2.58-GiB host LRU, not permanent target-radix
  ownership.
- Sparse small-M work returns directly in BF16. At 16 rows and wider, the four
  Sparks exchange FP8 row shards over Spark-RDMA and the owner returns BF16.
