# Legacy stream checksum fixtures

Files from the tablet writer as it was just before stream trailers
(`FileProperties.stream_trailer`), when per-stream checksums lived in
stripe-group arrays. TabletTest reads them to check that current readers still
decode those arrays. The current writer cannot produce such files, so never
regenerate them from it; write any new ones at a revision before stream
trailers, with the generator below.

Each file was written by `TabletWriter` with `streamChecksumsEnabled = true`,
`streamDeduplicationEnabled = false` and the stripe-group layout in its name,
plus the `columnar.properties` section `nimble::Writer` added for such files:
`FileProperties{false, false, {}, /*hasStreamChecksums=*/true}`. Each stream's
bytes count up from a first byte of its own, so no two streams are alike.

### `legacy_stripe_group_checksums_{raw,stream_major}.nimble`

Two stripes in one stripe group. Stripe 0 (50 rows) has streams 0, 1 and 2 of
10, 20 and 8 + 12 bytes, stream 2 in two chunks. Stripe 1 (30 rows) has streams
0 and 1 of 15 and 25 bytes.

### `legacy_stripe_group_checksums_multi_group_{raw,stream_major}.nimble`

Twelve 10-row stripes of three streams each; stream `s` of stripe `t` holds
`10 + 3t + s` bytes. `metadataFlushThreshold = 100` splits them into six stripe
groups.

## Generator

Run as a test in `TabletTest.cpp` at that revision; it writes the files to
`/tmp/legacy_fixtures/`.

```cpp
nimble::Stream makeFixtureStream(
    nimble::Buffer& buffer,
    uint32_t streamId,
    uint32_t seed,
    const std::vector<std::pair<uint32_t, uint32_t>>& chunks) {
  nimble::Stream stream{.offset = streamId};
  auto value = static_cast<uint8_t>(seed * 37 + 1);
  for (const auto& [rowCount, size] : chunks) {
    auto* pos = buffer.reserve(size);
    for (uint32_t i = 0; i < size; ++i) {
      pos[i] = static_cast<char>(value++);
    }
    stream.chunks.push_back({.rowCount = rowCount, .content = {{pos, size}}});
  }
  return stream;
}

TEST_P(TabletTest, generateLegacyFixtures) {
  const auto properties =
      nimble::FileProperties{false, false, {}, /*hasStreamChecksums=*/true}
          .serialize();
  const auto save = [](const std::string& name, const std::string& data) {
    std::ofstream out("/tmp/legacy_fixtures/" + name, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  };
  for (const auto layout :
       {nimble::StripeGroup::EncodingLayout::kRaw,
        nimble::StripeGroup::EncodingLayout::kStreamMajor}) {
    const std::string suffix =
        layout == nimble::StripeGroup::EncodingLayout::kRaw ? "raw"
                                                            : "stream_major";
    {
      std::string file;
      velox::InMemoryWriteFile writeFile(&file);
      auto tabletWriter = nimble::TabletWriter::create(
          &writeFile,
          *pool_,
          {.streamChecksumsEnabled = true,
           .streamDeduplicationEnabled = false,
           .stripeGroupEncodingLayout = layout});
      nimble::Buffer buffer{*pool_};
      {
        std::vector<nimble::Stream> streams;
        streams.push_back(makeFixtureStream(buffer, 0, 0, {{50, 10}}));
        streams.push_back(makeFixtureStream(buffer, 1, 1, {{50, 20}}));
        streams.push_back(
            makeFixtureStream(buffer, 2, 2, {{25, 8}, {25, 12}}));
        tabletWriter->writeStripe(50, std::move(streams));
      }
      {
        std::vector<nimble::Stream> streams;
        streams.push_back(makeFixtureStream(buffer, 0, 3, {{30, 15}}));
        streams.push_back(makeFixtureStream(buffer, 1, 4, {{30, 25}}));
        tabletWriter->writeStripe(30, std::move(streams));
      }
      tabletWriter->writeOptionalSection(
          std::string(nimble::kPropertiesSection), properties);
      tabletWriter->close();
      writeFile.close();
      save("legacy_stripe_group_checksums_" + suffix + ".nimble", file);
    }
    {
      std::string file;
      velox::InMemoryWriteFile writeFile(&file);
      auto tabletWriter = nimble::TabletWriter::create(
          &writeFile,
          *pool_,
          {.metadataFlushThreshold = 100,
           .streamChecksumsEnabled = true,
           .streamDeduplicationEnabled = false,
           .stripeGroupEncodingLayout = layout});
      nimble::Buffer buffer{*pool_};
      for (uint32_t stripe = 0; stripe < 12; ++stripe) {
        std::vector<nimble::Stream> streams;
        for (uint32_t s = 0; s < 3; ++s) {
          const uint32_t size = 10 + stripe * 3 + s;
          streams.push_back(
              makeFixtureStream(buffer, s, 5 + stripe * 3 + s, {{10, size}}));
        }
        tabletWriter->writeStripe(10, std::move(streams));
      }
      tabletWriter->writeOptionalSection(
          std::string(nimble::kPropertiesSection), properties);
      tabletWriter->close();
      writeFile.close();
      save(
          "legacy_stripe_group_checksums_multi_group_" + suffix + ".nimble",
          file);
    }
  }
}
```
