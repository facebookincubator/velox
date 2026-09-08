# cuDF Parquet split batching

Submit original splits together using the vector overload:

```cpp
std::vector<exec::Split> splits;
splits.emplace_back(std::move(firstFileSplit));
splits.emplace_back(std::move(secondFileSplit));
task->addSplit(scanNodeId, std::move(splits));
```

Task preserves the vector as one scheduled batch for connectors that enable
`supportsSplitBatch()`. TableScan passes the original connector splits to
`DataSource::addSplit(vector<shared_ptr<ConnectorSplit>>)`. Other connectors
receive individually queued splits. Existing single-split APIs remain supported.
Grouped, barrier, and mixed-connector submissions use individual delivery.

The queue uses a generic `ConnectorSplitBatch` internally to retain each child's
metadata and the total split weight. Callers do not construct this wrapper.
Task and operator split counts count scheduled batches; the original files are
available through reader counters. Trace capture writes the original splits.

## cuDF reader

cuDF connectors enable batch delivery when `cudf.batch_splits_enabled=true`.
With `cudf.hive.use-experimental-reader=true`, compatible complete Parquet files
initialize one `cudf::io::parquet::experimental::hybrid_scan_multifile` with all
of their metadata. Single-file splits use `hybrid_scan_reader`, including its
byte-range filtering. The regular reader supports compatible file vectors
through `chunked_parquet_reader`.

Row-group statistics pruning returns column ranges with source indices. Velox
fetches each source through its existing datasource and restores cuDF's flattened
range order. Fetched device buffers live until the reader is destroyed. Chunk and
pass limits bound decoding/output; they do not cap compressed input retained by a
batch. Callers must bound their vectors and submit enough batches to occupy their
scan drivers. Task does not regroup or resize a vector.

The datasource checks whole-file coverage against actual file sizes. Incompatible
splits retain their original per-file processing and metadata. The existing cuDF
file-list split constructors also remain available for direct reader callers.

## Integration and observation

Presto collects the splits accepted from each TaskSource update into a vector
and calls Task's vector overload. It reads Task's existing sequence watermark to
exclude retried deliveries before submission, then advances that watermark.
Delivery remains serialized by the existing Presto task lock. There is no
Presto cuDF batcher, duplicate sequence state, or connector-specific grouping
policy. Each update defines its batch boundary; there is no worker byte cap.

Scan runtime statistics expose `parquet.cudfMultiFileReaders` and
`parquet.cudfBatchedFiles`. They count actual hybrid multi-file reader
constructions and their input files, including files subsequently pruned by
statistics. Batches that fall back per file do not increment them.
