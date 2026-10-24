# Legacy stream checksum fixtures

Files from `nimble::Writer` as it was just before stream trailers
(`FileProperties.stream_trailer`), when per-stream checksums lived in
stripe-group arrays and the properties set `has_stream_checksums`.
NimbleIndexProjectorTest verifies projections from them. The current writer
cannot produce such files, so never regenerate them from it; write any new ones
at a revision before stream trailers, with the generator below.

### `legacy_indexed_stripe_group_checksums_{raw,stream_major}.nimble`

400 rows of `{key BIGINT, valueA INTEGER, valueB INTEGER, valueC INTEGER}`: row
`r` holds `key = r * 10`, `valueA = valueB = r * 100` and `valueC = r * 7`,
so valueA and valueB deduplicate. Four 100-row stripes in two stripe groups,
with a cluster index on `key`, Zstd chunk compression, no user metadata and the
stripe-group layout in the file name.

## Generator

Run as a test in `NimbleIndexProjectorTest.cpp` at that revision; it writes
the files to `/tmp/legacy_fixtures/`.

```cpp
TEST_P(NimbleIndexProjectorChecksumTest, generateLegacyFixtures) {
  std::vector<RowVectorPtr> batches;
  for (int batch = 0; batch < 4; ++batch) {
    std::vector<int64_t> keys(100);
    std::vector<int32_t> values(100);
    std::vector<int32_t> others(100);
    for (int i = 0; i < 100; ++i) {
      const int row = batch * 100 + i;
      keys[i] = row * 10;
      values[i] = row * 100;
      others[i] = row * 7;
    }
    batches.push_back(vectorMaker_->rowVector(
        {"key", "valueA", "valueB", "valueC"},
        {vectorMaker_->flatVector<int64_t>(keys),
         vectorMaker_->flatVector<int32_t>(values),
         vectorMaker_->flatVector<int32_t>(values),
         vectorMaker_->flatVector<int32_t>(others)}));
  }
  for (const auto layout :
       {StripeGroup::EncodingLayout::kRaw,
        StripeGroup::EncodingLayout::kStreamMajor}) {
    WriterOptions options;
    options.metadata = {};
    options.metadataFlushThreshold = 100;
    options.enableStreamChecksums = true;
    options.enableStreamDeduplication = true;
    options.enableChunking = true;
    options.experimentalStripeGroupEncodingLayout = layout;
    options.chunkCompression = {
        .type = CompressionType::Zstd, .acceptRatio = 10.0f};
    options.clusterIndexConfig = makeClusterIndexConfig({"key"});
    options.flushPolicyFactory = []() {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress& progress) {
            return progress.stripeRawSize >= (1 << 10);
          });
    };
    writeBatches(batches, std::move(options));
    const std::string path = fmt::format(
        "/tmp/legacy_fixtures/legacy_indexed_stripe_group_checksums_{}.nimble",
        layout == StripeGroup::EncodingLayout::kRaw ? "raw" : "stream_major");
    std::ofstream out(path, std::ios::binary);
    out.write(sinkData_.data(), static_cast<std::streamsize>(sinkData_.size()));
  }
}
```
