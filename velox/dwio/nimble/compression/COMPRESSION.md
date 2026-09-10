# Nimble Compression Guide

This document explains how compression works in Nimble, how to adjust compression settings, and how to add a new compression codec.

## Architecture Overview

Nimble's compression system has three layers:

1. **Codec layer** — individual compressor implementations (`ICompressor`) that perform the actual compress/decompress work.
2. **Policy layer** — `CompressionPolicy` decides *which* codec to use, with what parameters, and whether to accept the result.
3. **Configuration layer** — `CompressionOptions` (exposed via `WriterOptions`) lets users tune compression behavior.

Compression is applied at **leaf encoding nodes** in the encoding tree (e.g. `Trivial`, `FixedBitWidth`). The `CompressionEncoder` template class in `Compression.h` handles the common pattern of trying compression and falling back to uncompressed if the policy rejects the result.

```
WriterOptions::compressionOptions
  → ManualEncodingSelectionPolicy (encoding selection)
    → ConfiguredCompressionPolicy (created per-stream at leaf level)
      → Compression::compress() (static dispatch)
        → CompressorRegistry → ICompressor::compress()
```

## Supported Codecs

| Codec | Enum Value | Class | Files | Default Settings | Min Size | OSS |
|-------|-----------|-------|-------|-----------------|----------|-----|
| Uncompressed | 0 | — | — | — | — | Yes |
| Zstd | 1 | `ZstdCompressor` | `ZstdCompressor.{h,cpp}` | level=3 | 25 bytes | Yes |
| MetaInternal | 2 | `MetaInternalCompressor` | `fb/MetaInternal{,MC}Compressor.{h,cpp}` | comp=4, decomp=2 | 40 bytes | No |
| LZ4 | 3 | `Lz4Compressor` | `Lz4Compressor.{h,cpp}` | accel=1 | 12 bytes | Yes |

The default codec is **MetaInternal** internally and **Zstd** in OSS builds (controlled by `DISABLE_META_INTERNAL_COMPRESSOR`).

## Adjusting Compression Settings

### Option 1: Direct `WriterOptions` (programmatic)

Set fields on `WriterOptions::compressionOptions` before constructing a `Writer`:

```cpp
#include "velox/dwio/nimble/writer/WriterOptions.h"

WriterOptions options;

// Switch to Zstd with higher compression level
options.compressionOptions.compressionType = CompressionType::Zstd;
options.compressionOptions.zstdCompressionLevel = 7;

// Reject compressed output unless it saves at least 5%
options.compressionOptions.compressionAcceptRatio = 0.95f;

// Increase minimum stream size before attempting compression
options.compressionOptions.zstdMinCompressionSize = 64;

Writer writer(type, std::move(file), pool, std::move(options));
```

### Option 2: Serde Parameters (Hive table properties)

When writing through the DWIO API layer (`NimbleWriterOptionBuilder`), compression settings are read from Hive serde parameters:

| Serde Parameter | Field |
|---|---|
| `alpha.encodingselection.compression.accept.ratio` | `compressionAcceptRatio` |
| `alpha.zstd.compression.size.min` | `zstdMinCompressionSize` |
| `alpha.zstrong.compression.size.min` | `internalMinCompressionSize` |
| `alpha.zstrong.compression.level` | `internalCompressionLevel` |
| `alpha.zstrong.decompression.level` | `internalDecompressionLevel` |
| `alpha.zstrong.enable.variable.bit.width.compressor` | `useVariableBitWidthCompressor` |
| `alpha.openzl.managed.compression.key` | `metaInternalCompressionKey` |

These are defined in `dwio/api/NimbleConfig.{h,cpp}`.

### Option 3: `WriterOptionOverrides` (high-level DWIO FileWriter)

The `WriterOptionOverrides` lambda lets you override options that were already populated from serde parameters:

```cpp
#include "dwio/api/FileWriter.h"

WriterOptionOverrides overrides;
overrides.nimbleOverrides = [](nimble::WriterOptions& opts) {
  opts.compressionOptions.compressionType = CompressionType::Lz4;
  opts.compressionOptions.lz4AccelerationLevel = 2;
};
```

### `CompressionOptions` Reference

Defined in `velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h`:

```cpp
struct CompressionOptions {
  // Reject compression if compressedSize > uncompressedSize * ratio
  float compressionAcceptRatio = 0.98f;

  // Which codec to use
  CompressionType compressionType = CompressionType::MetaInternal; // Zstd in OSS

  // Zstd settings
  uint64_t zstdMinCompressionSize = 25;   // skip streams smaller than this
  uint32_t zstdCompressionLevel = 3;       // zstd level (1=fast, 22=max)

  // LZ4 settings
  uint64_t lz4MinCompressionSize = 12;
  uint32_t lz4AccelerationLevel = 1;       // higher = faster but less compression

  // MetaInternal (Zstrong) settings
  uint64_t internalMinCompressionSize = 40;
  uint32_t internalCompressionLevel = 4;
  uint32_t internalDecompressionLevel = 2;
  bool useVariableBitWidthCompressor = false;
  MetaInternalCompressionKey metaInternalCompressionKey;
};
```

### Metadata Compression

Separate from per-stream data compression, metadata sections (stripe groups, optional sections) in the file footer are compressed with Zstd when they exceed `metadataCompressionThreshold` (default: 64 KB). Configure via `WriterOptions::metadataCompressionThreshold`.

## How Compression Selection Works

Compression is **not** per-column — a single `CompressionOptions` applies uniformly to all streams. The per-stream decision is accept/reject:

1. If the stream's raw size is below `minCompressionSize`, skip compression.
2. Compress the data with the configured codec.
3. Call `CompressionPolicy::shouldAccept()` — if `compressedSize > uncompressedSize * compressionAcceptRatio`, reject and keep uncompressed.
4. Individual codecs also reject internally (e.g. Zstd returns `Uncompressed` if `ZSTD_compress` produces output larger than input).

## Adding a New Compression Codec

### Step 1: Add the enum value

Add a new entry to `CompressionType` in **both** locations. The values must match.

**`velox/dwio/nimble/common/Types.h`:**
```cpp
enum class CompressionType : uint8_t {
  Uncompressed = 0,
  Zstd = 1,
  MetaInternal = 2,
  Lz4 = 3,
  MyCodec = 4,  // <-- new
};
```

**`velox/dwio/nimble/tablet/Footer.fbs`:**
```fbs
enum CompressionType:uint8 {
  Uncompressed = 0,
  Zstd = 1,
  MetaInternal = 2,
  Lz4 = 3,
  MyCodec = 4,
}
```

Also update the `toString(CompressionType)` function in `velox/dwio/nimble/common/Types.cpp`.

### Step 2: Implement the `ICompressor` interface

Create `MyCodecCompressor.h` and `MyCodecCompressor.cpp` in `velox/dwio/nimble/compression/`. The interface is defined in `Compression.h`:

```cpp
#pragma once

#include "velox/dwio/nimble/compression/Compression.h"

namespace facebook::nimble {

class MyCodecCompressor : public ICompressor {
 public:
  /// Compress data. Return CompressionType::Uncompressed with std::nullopt
  /// buffer if compression is not beneficial.
  CompressionResult compress(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      DataType dataType,
      int bitWidth,
      const CompressionPolicy& compressionPolicy) override;

  /// Decompress data. The compressionType parameter identifies which codec
  /// produced the data.
  velox::BufferPtr uncompress(
      velox::memory::MemoryPool& pool,
      CompressionType compressionType,
      DataType dataType,
      std::string_view data,
      velox::BufferPool* bufferPool = nullptr) override;

  /// Return the uncompressed size from the compressed data header, if
  /// available.
  std::optional<size_t> uncompressedSize(std::string_view data) const override;

  CompressionType compressionType() override;
};

} // namespace facebook::nimble
```

Implementation notes (follow the pattern in `ZstdCompressor.cpp` or `Lz4Compressor.cpp`):

- **Prepend the uncompressed size** as a `uint32_t` header before the compressed payload. Both Zstd and LZ4 do this so `uncompressedSize()` can read it without decompressing.
- **Return `Uncompressed`** if the compressed output is not smaller than the input — return `{CompressionType::Uncompressed, std::nullopt}`.
- **Call `compressionPolicy.shouldAccept()`** to let the policy decide whether the compression ratio is good enough. If it returns `false`, return `Uncompressed`.
- **Define a minimum compression size constant** in `velox/dwio/nimble/common/Constants.h` (e.g. `constexpr uint64_t kMyCodecMinCompressionSize = 20;`).
- Use `allocateBuffer()` from `Compression.h` to allocate output buffers.

### Step 3: Register the compressor

Add the new compressor to the `CompressorRegistry` in `Compression.cpp`:

```cpp
#include "velox/dwio/nimble/compression/MyCodecCompressor.h"

struct CompressorRegistry {
  CompressorRegistry() {
    compressors.reserve(4);  // <-- update count
    compressors.emplace(CompressionType::Zstd, std::make_unique<ZstdCompressor>());
    compressors.emplace(CompressionType::Lz4, std::make_unique<Lz4Compressor>());
    compressors.emplace(CompressionType::MyCodec, std::make_unique<MyCodecCompressor>());  // <-- new
#ifndef DISABLE_META_INTERNAL_COMPRESSOR
    compressors.emplace(CompressionType::MetaInternal, std::make_unique<MetaInternalCompressor>());
#endif
  }
  // ...
};
```

### Step 4: Add compression parameters

**`velox/dwio/nimble/compression/CompressionPolicy.h`** — add a parameter struct and include it in `CompressionParameters`:

```cpp
struct MyCodecCompressionParameters {
  int16_t myOption = 5;
};

struct CompressionParameters {
  ZstdCompressionParameters zstd{};
  Lz4CompressionParameters lz4{};
  MetaInternalCompressionParameters metaInternal{};
  MyCodecCompressionParameters myCodec{};  // <-- new
};
```

### Step 5: Add configuration fields

**`velox/dwio/nimble/compression/CompressionPolicy.h`** — add fields to `CompressionOptions`:

```cpp
struct CompressionOptions {
  // ... existing fields ...
  uint64_t myCodecMinCompressionSize = kMyCodecMinCompressionSize;
  int32_t myCodecOption = 5;
};
```

**`velox/dwio/nimble/compression/CompressionPolicy.cpp`** — update `ConfiguredCompressionPolicy::config()` to handle the new codec:

```cpp
CompressionConfig config() const override {
  // ... existing Zstd and Lz4 branches ...
  if (compressionOptions_.compressionType == CompressionType::MyCodec) {
    CompressionConfig information{
        .compressionType = CompressionType::MyCodec,
        .minCompressionSize = compressionOptions_.myCodecMinCompressionSize};
    information.parameters.myCodec.myOption =
        compressionOptions_.myCodecOption;
    return information;
  }
  // ... MetaInternal fallthrough ...
}
```

### Step 6: Update the BUCK file

Add the new source and header files to `velox/dwio/nimble/compression/BUCK`, plus any third-party dependency:

```python
cpp_library(
    name = "compression",
    srcs = [
        "Compression.cpp",
        "Lz4Compressor.cpp",
        "MyCodecCompressor.cpp",  # <-- new
        "ZstdCompressor.cpp",
        "fb/MetaInternalCompressor.cpp",
        "fb/MetaInternalMCCompressor.cpp",
    ],
    headers = [
        # ... existing headers ...
        "MyCodecCompressor.h",  # <-- new
    ],
    deps = [
        # ... existing deps ...
        "fbsource//third-party/mycodec:mycodec",  # <-- if needed
    ],
    # ...
)
```

### Step 7 (optional): Wire up serde parameters

If the codec should be configurable via Hive table properties, add config entries in `dwio/api/NimbleConfig.{h,cpp}` and populate them in `dwio/api/NimbleWriterOptionBuilder.cpp`.

### Step 8: Add tests

Add compression round-trip tests in `velox/dwio/nimble/encodings/tests/` or `velox/dwio/nimble/compression/tests/`. At minimum, verify:

- Compress → decompress round-trip produces identical data.
- Streams below `minCompressionSize` are not compressed.
- Streams that don't compress well are kept uncompressed.
- `uncompressedSize()` returns the correct value from the header.
- End-to-end: write a Nimble file with the new codec and read it back.

## Checklist for Adding a New Codec

- [ ] Add enum value to `CompressionType` in `Types.h` and `Footer.fbs` (must match)
- [ ] Update `toString(CompressionType)` in `Types.cpp`
- [ ] Implement `ICompressor` subclass (`MyCodecCompressor.{h,cpp}`)
- [ ] Add min compression size constant to `Constants.h`
- [ ] Register in `CompressorRegistry` in `Compression.cpp`
- [ ] Add parameter struct to `CompressionPolicy.h`
- [ ] Add fields to `CompressionOptions` in `CompressionPolicy.h`
- [ ] Handle new codec in `ConfiguredCompressionPolicy::config()`
- [ ] Add source/header to `compression/BUCK`
- [ ] (Optional) Add serde parameters in `NimbleConfig.{h,cpp}` and `NimbleWriterOptionBuilder.cpp`
- [ ] Add unit tests

## Key Files Reference

| File | Purpose |
|------|---------|
| `common/Types.h` | `CompressionType` enum |
| `common/Constants.h` | Min compression size constants |
| `compression/Compression.h` | `ICompressor` interface, `CompressionEncoder`, `Compression` static API |
| `compression/Compression.cpp` | `CompressorRegistry`, dispatch logic |
| `compression/CompressionPolicy.h` | `CompressionPolicy` interface, parameter structs, compression options |
| `compression/CompressionPolicy.cpp` | `ConfiguredCompressionPolicy` implementation |
| `compression/ZstdCompressor.{h,cpp}` | Zstd codec (good reference implementation) |
| `compression/Lz4Compressor.{h,cpp}` | LZ4 codec |
| `compression/fb/MetaInternalCompressor.{h,cpp}` | Zstrong/Managed Compression (internal only) |
| `compression/BUCK` | Build target for all compression code |
| `encodings/selection/EncodingSelectionPolicy.h` | Encoding selection policy |
| `velox/WriterOptions.h` | `WriterOptions::compressionOptions` |
| `tablet/Footer.fbs` | FlatBuffers schema (stores `CompressionType` per metadata section) |
| `dwio/api/NimbleConfig.{h,cpp}` | Serde parameter definitions |
| `dwio/api/NimbleWriterOptionBuilder.cpp` | Populates `CompressionOptions` from serde params |
