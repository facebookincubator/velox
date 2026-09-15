# GPU packed-column compression

This directory provides transport-neutral compression for GPU-resident
`cudf::packed_columns`. It has no dependency on UCX, network endpoints, query
stages, or exchange server state.

## Public API

`PackedColumnsCodec.h` is the public API. `PackedColumnsCodec` compresses the
GPU buffer produced by `cudf::pack`, serializes an opaque descriptor, and
reconstructs the buffer byte-exactly before `cudf::unpack`. The FOR,
delta-FOR, and ANS stages are implementation details related to the compression
algorithm.

The codec accepts the owning `cudf::packed_columns` object, so its metadata,
device pointer, and size cannot disagree. Compressed inputs are device spans,
which likewise keep their pointer and size together.

A codec instance belongs to one CUDA stream and has explicit temporary and
output memory resources. It reuses its nvCOMP manager and pinned host staging
across calls. Calls are synchronous with respect to that stream, and the codec
must be destroyed before its stream.

The codec automatically chooses an encoding for each region. It has no
link-rate or UCX state. When invoked, it rejects output that does not meet its
minimum byte-reduction safeguard. The consumer decides whether attempting
compression is expected to improve end-to-end runtime.

## Build option

Set `VELOX_ENABLE_CUDF_COMPRESSION=ON` to build the standalone target when
UCX exchange is disabled. Enabling UCX exchange builds this required consumer
dependency automatically. Ordinary cuDF builds with both options disabled do
not compile the codec. The layer uses the nvCOMP dependency already supplied by
the cuDF build.

## Consumer boundary

UCX is one possible consumer. A transport adapter owns transfer-rate
observations, adaptive decisions, endpoint lifecycle, and query-stage policy.
Cache and spill consumers can use the same codec API with their own policy and
lifetime rules.

The `cudf_compression_test` target links this library without UCX. Its
round-trip tests cover logical numeric types, decimals, timestamps, null masks,
nested strings, extreme integer values, malformed descriptors, input bounds,
and byte-exact reconstruction.
