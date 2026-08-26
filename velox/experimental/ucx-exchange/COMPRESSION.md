# GPU compression for UCX exchange

This optional path compresses cuDF exchange buffers on the sending GPU and
reconstructs them on the receiving GPU before `cudf::unpack`. Its purpose is
to reduce inter-worker traffic when the saved transfer time is greater than the
GPU codec cost.

Compression is disabled by default. All workers in a query must use a build
with the same wire-format support.

## Scope

This branch contains the numeric codec stack, adaptive selection, transport
gating, and UCX send/receive integration. It intentionally excludes
Parquet/storage codecs, cross-column research, string-prefix experiments, and
standalone benchmarks.

## Data path

The sender receives the normal `cudf::packed_columns` output. It leaves the
cuDF host metadata unchanged and divides the contiguous GPU data buffer into
regions:

1. Fixed-width numeric regions are candidates for FOR, delta-FOR,
   dictionary-PFOR, frequency-PFOR, or delta-frequency-PFOR.
2. The transformed byte planes and residual regions are compressed where useful
   with DietGPU's byte-oriented rANS codec. Dictionary-PFOR rank bytes are sent
   directly.
3. Unsupported or unprofitable regions are copied raw.
4. A compact region descriptor is placed in the existing UCX metadata message.
5. The receiver decodes each region into a byte-exact copy of the original GPU
   buffer and then calls the unchanged `cudf::unpack` path.

ANS is the entropy-coding family. DietGPU implements the range variant, rANS.
The code therefore uses DietGPU ANS and may refer to the concrete coder as
rANS. FOR and PFOR are transforms that expose lower-entropy byte planes before
rANS, not competing entropy-coder libraries.

The build fetches a pinned official DietGPU source archive and verifies its
SHA-256 checksum. Only the byte-rANS sources required by this path are compiled.

## Codec selection

The column codec evaluates candidates on the GPU and keeps a compressed region
only when it is smaller than the raw region. The complete compressed chunk must
save at least two percent or the original buffer is sent.

The recommended adaptive mode also maintains a small online model per query
stage. Its first samples are normal encodes that are sent, not duplicate probe
passes. It combines measured encode time, UCX send-completion time, decode
time, and compression ratio. Compression is selected only when the estimated
transfer saving exceeds the measured codec cost by the configured safety
margin. Stages that select raw transfer are periodically reprobed.

Broadcast destinations that share the same packed buffer also share one
compression result. The codec runs on a bounded executor instead of the UCXX
progress thread so UCX can continue making progress. One codec thread per
worker is the conservative starting point because one codec operation can
already use much of the GPU.

Compression is never used for the in-process same-worker path. UCP endpoint
transport discovery also leaves CUDA IPC transfers raw. A known non-CUDA-IPC
endpoint, such as a remote TCP or RDMA endpoint, is eligible. If UCP cannot
report the endpoint transports, the implementation fails closed and sends raw
data.

## Configuration

The worker's native configuration accepts these properties:

| Property | Default | Meaning |
| --- | --- | --- |
| `cudf.exchange_compression` | `none` | Codec policy described below. |
| `cudf.exchange_compression_pipeline` | `false` | Run encode and decode outside the UCXX progress thread. |
| `cudf.exchange_compression_pipeline_threads` | `1` | Bounded codec executor size, clamped to 1 through 4. |
| `cudf.exchange_compression_min_bytes` | `0` | Do not probe chunks smaller than this many bytes. |
| `cudf.exchange_compression_safety_margin` | `1.10` | Required transfer-saving to codec-cost ratio in adaptive modes. |

The main policies are:

| Value | Behavior |
| --- | --- |
| `none` | Send the original GPU buffer. |
| `ans` | Apply DietGPU byte-rANS to fixed-size segments of the whole buffer. This is mainly a diagnostic baseline. |
| `column` | Always try the basic per-column FOR, delta-FOR, byte-rANS, and raw candidates. |
| `column-adaptive` | Use the online cost model with the basic per-column candidates. |
| `column-adaptive-freq-pfor-min128` | Use the online cost model and enable dictionary-PFOR, frequency-PFOR, and delta-frequency-PFOR candidates only on numeric regions of at least 128 MiB. |

The implementation retains non-adaptive and legacy PFOR policy names for
controlled comparisons. The `*-freq-pfor-min128` policy avoids paying the
advanced histogram and dictionary setup cost on small regions.

A reasonable starting configuration for a remote GPU link is:

```properties
cudf.exchange=true
cudf.exchange_compression=column-adaptive-freq-pfor-min128
cudf.exchange_compression_pipeline=true
cudf.exchange_compression_pipeline_threads=1
cudf.exchange_compression_min_bytes=16777216
cudf.exchange_compression_safety_margin=1.50
```

These values are intentionally conservative, not universal. Link bandwidth,
GPU contention, exchange chunk sizes, and data distributions change the
break-even point. Measure both compressed and raw exchange on the target
system. Setting `cudf.exchange_compression=none` restores the original data
path without removing UCX.

## Build and validation

Build Velox with cuDF, UCX exchange, and CUDA enabled. The UCX exchange CMake
target downloads the pinned DietGPU archive during configuration. A
network-isolated build must provide the archive through the normal CMake
FetchContent cache.

With `VELOX_BUILD_TESTING=ON`, build the focused target:

```bash
cmake --build _build/release -j4 --target ucx_exchange_test
```

Run the codec and cost-model tests on a CUDA-capable host:

```bash
_build/release/velox/experimental/ucx-exchange/tests/ucx_exchange_test \
  --gtest_filter='UcxCompressionTest.*:UcxCompressionCostModelTest.*'
```

The GPU tests cover skipped inputs, whole-buffer rANS round trips, descriptor
round trips, frequency-PFOR exception patching, delta reconstruction, and
byte-exact output. The CPU-only cost-model tests cover warmup, selection,
safety margin, periodic reprobes, and stage-local/global sampling.

For an end-to-end deployment, additionally verify:

1. The worker log shows UCX exchange rather than HTTP exchange.
2. Every worker uses this wire format.
3. The selected endpoint is reported as non-CUDA-IPC before compression occurs.
4. Compressed and raw query runs return identical results.
5. Logs report both lower wire bytes and a favorable end-to-end runtime or a
   justified traffic reduction.

## Implementation map

- `UcxColumnCodec.*`: region discovery, numeric transforms, rANS residuals,
  descriptor serialization, and byte-exact reconstruction.
- `UcxCompression.*`: segmented whole-buffer DietGPU rANS wrapper.
- `UcxCompressionCostModel.*`: adaptive stage-level selection.
- `UcxCodecPipeline.h`: bounded off-progress-thread executor.
- `UcxExchangeServer.*`: transport gating, encode, broadcast reuse, and send.
- `UcxExchangeSource.*`: descriptor parsing, decode, and enqueue.
- `EndpointRef.*`: cached UCP transport discovery.

This is an experimental wire extension. It does not provide mixed-version
compatibility with workers that lack the descriptor decoder.
