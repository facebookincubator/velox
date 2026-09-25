/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"

// ---------------------------------------------------------------------------
// Addressable block compression
// ---------------------------------------------------------------------------
// OuterCompressedTarget (BenchCommon.h) wraps a whole encoded column in one
// block codec, so every read there decompresses everything. This file is the
// addressable sibling: a columnar format ships a codec in fixed-size blocks
// and decompresses only the blocks a read overlaps, making read cost a
// function of block size rather than column size.

namespace facebook::nimble::mlidc {

/// Compresses and decompresses one fixed-size block of elements.
///
/// Exists so the block layout, addressing and accounting are written once and
/// every codec plugs into them, rather than each growing its own indexing.
template <typename T>
class BlockCodec {
 public:
  virtual ~BlockCodec() = default;

  /// Compresses `count` elements at `src`, appending the codec's bytes to
  /// `out`. Returns false when the codec declined the block, having appended
  /// nothing, in which case the caller stores the block verbatim.
  virtual bool
  compressBlock(const T* src, uint32_t count, std::string& out) = 0;

  /// Decompresses one block previously produced by compressBlock into `dst`,
  /// which has room for `count` elements.
  virtual void
  decompressBlock(std::string_view block, uint32_t count, T* dst) = 0;
};

/// BlockCodec backed by nimble's own compressor registry, so Zstd here is the
/// same Zstd the encodings use for their sub-streams.
template <typename T>
class NimbleBlockCodec : public BlockCodec<T> {
 public:
  explicit NimbleBlockCodec(CompressionType compressionType)
      : compressionType_{compressionType} {}

  bool compressBlock(const T* src, uint32_t count, std::string& out) override {
    const std::string_view raw{
        reinterpret_cast<const char*>(src),
        static_cast<size_t>(count) * sizeof(T)};
    BenchCompressPolicy policy{compressionType_};
    auto result = Compression::compress(
        *pool_,
        raw,
        TypeTraits<T>::dataType,
        /*bitWidth=*/static_cast<int>(sizeof(T) * 8),
        policy);
    if (!result.buffer.has_value()) {
      return false;
    }
    out.append(result.buffer->data(), result.buffer->size());
    return true;
  }

  void decompressBlock(std::string_view block, uint32_t count, T* dst)
      override {
    auto buffer = Compression::uncompress(
        *pool_,
        compressionType_,
        TypeTraits<T>::dataType,
        block,
        /*decompressCounter=*/nullptr);
    const size_t expected = static_cast<size_t>(count) * sizeof(T);
    NIMBLE_CHECK(
        buffer->size() >= expected,
        "Block decompressed short: expected {} bytes, got {}",
        expected,
        buffer->size());
    // nimble's compressor interface hands back an owned buffer rather than
    // writing into a caller-supplied one, so a block always costs one extra
    // copy here.
    std::memcpy(dst, buffer->template as<char>(), expected);
  }

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_{benchmarks::benchmarkPool()};
  CompressionType compressionType_;
};

/// Splits a column into fixed-size blocks, compresses each independently, and
/// serves a read by decompressing only the blocks that read overlaps.
///
/// A read of `count` elements starting at `begin` decompresses exactly the
/// blocks in [begin / K, (begin + count - 1) / K]. A point read therefore
/// costs one block whatever the column length is, which is the property this
/// arm exists to demonstrate.
template <typename T>
class BlockCompressedTarget : public NimbleBenchTargetBase<T> {
 public:
  /// Takes the codec to apply per block and the block size in elements.
  BlockCompressedTarget(
      std::unique_ptr<BlockCodec<T>> codec,
      uint32_t blockSize,
      std::string codecName)
      : codec_{std::move(codec)},
        blockSize_{blockSize},
        codecName_{std::move(codecName)} {
    NIMBLE_CHECK(blockSize_ > 0, "Block size must be positive");
  }

  void encode(const Vector<T>& data, const Encoding::Options&) override {
    count_ = data.size();
    payload_.clear();
    blocks_.clear();
    cachedBlock_ = kNoBlock;
    numBlockDecodes_ = 0;

    const uint32_t numBlocks = numBlocksFor(count_, blockSize_);
    blocks_.reserve(numBlocks);
    for (uint32_t block = 0; block < numBlocks; ++block) {
      const uint32_t begin = block * blockSize_;
      // The final block is short whenever count_ is not a multiple of
      // blockSize_, and is the whole column when count_ is below it.
      const uint32_t elements = std::min(blockSize_, count_ - begin);
      const size_t offset = payload_.size();
      const bool compressed =
          codec_->compressBlock(data.data() + begin, elements, payload_);
      if (!compressed) {
        payload_.append(
            reinterpret_cast<const char*>(data.data() + begin),
            static_cast<size_t>(elements) * sizeof(T));
      }
      blocks_.push_back(
          {.offset = offset,
           .size = payload_.size() - offset,
           .elements = elements,
           .compressed = compressed});
    }
    // metadataBytes() charges a 32-bit start offset per block, so a payload
    // that could not be addressed by one would under-report its stored size.
    NIMBLE_CHECK(
        payload_.size() <= std::numeric_limits<uint32_t>::max(),
        "Block payload too large for a 32-bit block directory: {} bytes",
        payload_.size());
  }

  void materializeAll(T* dst, uint32_t n) override {
    cachedBlock_ = kNoBlock;
    readRange(0, n, dst);
  }

  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    cachedBlock_ = kNoBlock;
    readRange(begin, count, dst);
  }

  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    // The scratch block is reused across the ranges of one gather, since two
    // ranges landing in the same block should cost one decompression. It is
    // dropped at both ends of the call so nothing is cached between calls.
    cachedBlock_ = kNoBlock;
    for (const auto& range : ranges) {
      readRange(range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
    cachedBlock_ = kNoBlock;
  }

  /// Stored bytes: the compressed blocks plus the block directory, since a
  /// reader cannot address a block without the directory.
  size_t payloadSize() const override {
    return payload_.size() + metadataBytes();
  }

  /// Bytes the block directory would occupy on disk: a 32-bit start offset
  /// and a stored-form byte per block, one terminating offset, and a header
  /// giving the element count and the block size. Computed rather than
  /// serialised, since nothing here reads a block by parsing bytes.
  size_t metadataBytes() const {
    return blocks_.size() * (sizeof(uint32_t) + 1) + sizeof(uint32_t) +
        2 * sizeof(uint32_t);
  }

  /// The compressed blocks, the block directory, and the one scratch block a
  /// partial read decompresses into: a block arm holds its column compressed
  /// and one block uncompressed.
  size_t residentBytes() const override {
    return payload_.size() + blocks_.size() * sizeof(BlockEntry) +
        scratch_.capacity() * sizeof(T);
  }

  /// Blocks decompressed since the last encode. Instrumentation for tests
  /// pinning the "a read decompresses only what it overlaps" property.
  size_t numBlockDecodes() const {
    return numBlockDecodes_;
  }

  size_t numBlocks() const {
    return blocks_.size();
  }

  /// Block-addressable, except when the whole column landed in one block.
  /// Derived from the encoded shape rather than declared, so a single-block
  /// arm cannot claim an addressability it does not have.
  ReadPath readPath() const override {
    return blocks_.size() <= 1 ? ReadPath::kWholePayload : ReadPath::kBlock;
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(payload_.data()), payload_.size()},
        {reinterpret_cast<const std::byte*>(blocks_.data()),
         blocks_.size() * sizeof(BlockEntry)}};
  }

  std::string describe() override {
    size_t stored = 0;
    for (const auto& block : blocks_) {
      stored += block.compressed ? 0 : 1;
    }
    std::ostringstream out;
    out << "BlockCodec codec=" << codecName_ << " blockElements=" << blockSize_
        << " elements=" << count_ << " blocks=" << blocks_.size()
        << " compressedBytes=" << payload_.size()
        << " metadataBytes=" << metadataBytes() << " blocksStoredRaw=" << stored
        << "\n";
    return out.str();
  }

 private:
  // A block's slice of payload_ and how it was stored.
  struct BlockEntry {
    size_t offset;
    size_t size;
    uint32_t elements;
    bool compressed;
  };

  static constexpr uint32_t kNoBlock = std::numeric_limits<uint32_t>::max();

  static uint32_t numBlocksFor(uint32_t count, uint32_t blockSize) {
    return static_cast<uint32_t>(
        (static_cast<size_t>(count) + blockSize - 1) / blockSize);
  }

  // Serves one range, touching only the blocks it overlaps.
  void readRange(uint32_t begin, uint32_t count, T* dst) {
    if (count == 0) {
      return;
    }
    NIMBLE_CHECK(
        static_cast<size_t>(begin) + count <= count_,
        "Read past end of column: begin {}, count {}, elements {}",
        begin,
        count,
        count_);

    const uint32_t end = begin + count;
    const uint32_t lastBlock = (end - 1) / blockSize_;
    for (uint32_t block = begin / blockSize_; block <= lastBlock; ++block) {
      const uint32_t blockBegin = block * blockSize_;
      const uint32_t elements = blocks_[block].elements;
      const uint32_t from = std::max(begin, blockBegin);
      const uint32_t to = std::min(end, blockBegin + elements);
      T* out = dst + (from - begin);

      if (from == blockBegin && to == blockBegin + elements) {
        // The range covers the block, so decode straight into the caller's
        // buffer: the bulk-scan path, with no extra copy.
        decodeBlock(block, out);
        continue;
      }
      if (cachedBlock_ != block) {
        // The block's own element count, not blockSize_: a whole-column arm
        // sets blockSize_ to a bound, where resizing to it would ask for 4
        // billion elements.
        scratch_.resize(elements);
        decodeBlock(block, scratch_.data());
        cachedBlock_ = block;
      }
      std::copy(
          scratch_.data() + (from - blockBegin),
          scratch_.data() + (to - blockBegin),
          out);
    }
  }

  void decodeBlock(uint32_t block, T* dst) {
    const auto& entry = blocks_[block];
    const std::string_view bytes{payload_.data() + entry.offset, entry.size};
    ++numBlockDecodes_;
    if (entry.compressed) {
      codec_->decompressBlock(bytes, entry.elements, dst);
      return;
    }
    std::memcpy(dst, bytes.data(), bytes.size());
  }

  std::unique_ptr<BlockCodec<T>> codec_;
  uint32_t blockSize_;
  std::string codecName_;
  uint32_t count_{0};
  // Every block's bytes back to back; blocks_ says where each one starts.
  std::string payload_;
  std::vector<BlockEntry> blocks_;
  std::vector<T> scratch_;
  uint32_t cachedBlock_{kNoBlock};
  size_t numBlockDecodes_{0};
};

/// Block sizes swept, in elements: brackets vector scale and row-group scale.
inline constexpr std::array<uint32_t, 3> kBlockElementCounts{
    1024,
    65'536,
    262'144};

/// Builds one arm for a codec at one block size. `codecName` becomes the
/// arm's prefix, so the codec and block size are both readable off the CSV's
/// encoding column. No commas: --mlidc_encoders splits its list on them.
template <typename T>
EncoderEntry<T> makeBlockCodecEntry(
    std::string codecName,
    std::string family,
    uint32_t blockSize,
    std::function<std::unique_ptr<BlockCodec<T>>()> makeCodec) {
  EncoderEntry<T> entry;
  const std::string variant = "block-" + std::to_string(blockSize);
  entry.name = codecName + "/" + variant;
  entry.family = std::move(family);
  entry.variant = variant;
  // A block is addressable, so a skip costs nothing, but reaching a row
  // inside one still decompresses the whole block.
  entry.isSequential = false;
  entry.fastSkip = false;
  entry.randomAccess = false;
  entry.factory = [blockSize, codecName, makeCodec = std::move(makeCodec)](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<BlockCompressedTarget<T>>(
        makeCodec(), blockSize, codecName);
    target->encode(data, opts);
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

/// The Zstd arms: a plotted baseline that previously appeared only as a
/// sub-stream codec inside other encodings.
template <typename T>
std::vector<EncoderEntry<T>> buildZstdBlockEncoders() {
  std::vector<EncoderEntry<T>> entries;
  entries.reserve(kBlockElementCounts.size());
  for (const uint32_t blockSize : kBlockElementCounts) {
    entries.push_back(
        makeBlockCodecEntry<T>(
            "zstd", "Zstd", blockSize, []() -> std::unique_ptr<BlockCodec<T>> {
              return std::make_unique<NimbleBlockCodec<T>>(
                  CompressionType::Zstd);
            }));
  }
  return entries;
}

/// Zstd over the whole column in one block: what a reader pays when it holds
/// a compressed column and no block directory. This is the arm the
/// materialise-on-first-access decorator has something to prove against.
/// Expressed as a single block rather than as a target of its own so the
/// codec, size accounting and indexing stay the ones the block arms use.
template <typename T>
EncoderEntry<T> buildZstdWholeEncoder() {
  auto entry = makeBlockCodecEntry<T>(
      "zstd",
      "Zstd",
      std::numeric_limits<uint32_t>::max(),
      []() -> std::unique_ptr<BlockCodec<T>> {
        return std::make_unique<NimbleBlockCodec<T>>(CompressionType::Zstd);
      });
  entry.name = "zstd/whole";
  entry.variant = "whole";
  entry.isSequential = true;
  return entry;
}

// ---------------------------------------------------------------------------
// Lazy partial materialisation
// ---------------------------------------------------------------------------

/// Keeps the blocks it has decoded, and only those.
///
/// MaterializingTarget decodes the entire column on first access, which makes
/// a fair comparison against a view impossible since a view materialises only
/// the sections that lack one. This is the middle case a reader actually
/// deploys: materialise at the granularity the format is addressable at, and
/// let a workload decide how much of the column ends up resident.
///
/// Wraps a block target rather than a whole-payload one on purpose: with a
/// whole-payload inner, decoding "one block" would decompress the entire
/// column and the laziness would be fictitious.
template <typename T>
class BlockLazyTarget : public NimbleBenchTargetBase<T> {
 public:
  /// Takes an inner block target the entry's factory has already encoded,
  /// the column length, and the block size that inner target was built with.
  /// The two block sizes must agree.
  BlockLazyTarget(
      std::unique_ptr<NimbleBenchTargetBase<T>> inner,
      uint32_t rowCount,
      uint32_t blockSize)
      : inner_{std::move(inner)}, rowCount_{rowCount}, blockSize_{blockSize} {
    NIMBLE_CHECK(blockSize_ > 0, "Block size must be positive");
    resetCache();
  }

  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    inner_->encode(data, opts);
    rowCount_ = data.size();
    resetCache();
  }

  /// Reads every row, bypassing the cache: if it populated the cache, one
  /// scan would leave the whole column resident and this arm would silently
  /// become the +materialize arm it exists to be distinguished from.
  void materializeAll(T* dst, uint32_t n) override {
    inner_->materializeAll(dst, n);
  }

  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    readRange(begin, count, dst);
  }

  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    for (const auto& range : ranges) {
      readRange(range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
  }

  /// Block-addressable, and it stays so however much has been cached: a read
  /// that misses still costs exactly the blocks it overlaps.
  ReadPath readPath() const override {
    return blockCount() <= 1 ? ReadPath::kWholePayload : ReadPath::kBlock;
  }

  /// False, deliberately: there is no one-time build to amortise here, since
  /// the cost is spread across the reads that happen to miss.
  bool buildsAccessStructure() const override {
    return false;
  }

  /// Drops every cached block, so the next read decodes again. Called inside
  /// the timed region that measures a read starting from nothing.
  void discardAccessStructure() override {
    resetCache();
  }

  size_t payloadSize() const override {
    return inner_->payloadSize();
  }

  /// The compressed column plus exactly the blocks a workload has decoded:
  /// starts at the inner target's footprint and rises towards full
  /// materialisation only as far as the reads actually reach.
  size_t residentBytes() const override {
    return inner_->residentBytes() + cachedBytes_ +
        cache_.capacity() * sizeof(std::vector<T>);
  }

  /// Blocks decoded since the last reset. Instrumentation for tests pinning
  /// "a k-probe workload touches min(k, blocks) blocks".
  size_t numBlockMaterializations() const {
    return numBlockMaterializations_;
  }

  /// Blocks currently held decoded.
  size_t numCachedBlocks() const {
    return numCachedBlocks_;
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return inner_->internalBuffers();
  }

  std::string describe() override {
    return inner_->describe();
  }

  std::string describeTree() override {
    return inner_->describeTree();
  }

  std::string describeNodeEstimates() override {
    return inner_->describeNodeEstimates();
  }

 private:
  uint32_t blockCount() const {
    return static_cast<uint32_t>(
        (static_cast<size_t>(rowCount_) + blockSize_ - 1) / blockSize_);
  }

  void resetCache() {
    cache_.assign(blockCount(), std::vector<T>{});
    cachedBytes_ = 0;
    numCachedBlocks_ = 0;
  }

  // Indexed by block number rather than looked up in a hash map, so a hit is
  // an array index and the bookkeeping stays out of the measured loop.
  void ensureBlock(uint32_t block) {
    if (!cache_[block].empty()) {
      return;
    }
    const uint32_t blockBegin = block * blockSize_;
    const uint32_t elements = std::min(blockSize_, rowCount_ - blockBegin);
    std::vector<T> decoded(elements);
    inner_->materializeRange(blockBegin, elements, decoded.data());
    cachedBytes_ += decoded.capacity() * sizeof(T);
    ++numCachedBlocks_;
    ++numBlockMaterializations_;
    cache_[block] = std::move(decoded);
  }

  void readRange(uint32_t begin, uint32_t count, T* dst) {
    if (count == 0) {
      return;
    }
    NIMBLE_CHECK(
        static_cast<size_t>(begin) + count <= rowCount_,
        "Read past end of column: begin {}, count {}, elements {}",
        begin,
        count,
        rowCount_);
    const uint32_t end = begin + count;
    const uint32_t lastBlock = (end - 1) / blockSize_;
    for (uint32_t block = begin / blockSize_; block <= lastBlock; ++block) {
      ensureBlock(block);
      const uint32_t blockBegin = block * blockSize_;
      const uint32_t from = std::max(begin, blockBegin);
      const uint32_t to = std::min(
          end, blockBegin + static_cast<uint32_t>(cache_[block].size()));
      std::copy(
          cache_[block].data() + (from - blockBegin),
          cache_[block].data() + (to - blockBegin),
          dst + (from - begin));
    }
  }

  std::unique_ptr<NimbleBenchTargetBase<T>> inner_;
  uint32_t rowCount_{0};
  uint32_t blockSize_{0};
  // One entry per block; an empty entry means that block is not decoded.
  std::vector<std::vector<T>> cache_;
  size_t cachedBytes_{0};
  size_t numCachedBlocks_{0};
  size_t numBlockMaterializations_{0};
};

/// Wraps a block arm so it keeps the blocks it decodes. Composed like
/// withMaterializedAccess so the pair encodes to the same bytes and differs
/// only in what a read leaves behind. blockSize must match the wrapped
/// entry's.
template <typename T>
EncoderEntry<T> withBlockLazyMaterialization(
    EncoderEntry<T> entry,
    uint32_t blockSize) {
  entry.name += "+lazy";
  entry.variant += "_lazy";
  entry.isSequential = false;
  entry.fastSkip = true;
  entry.randomAccess = true;
  auto inner = std::move(entry.factory);
  entry.factory = [inner = std::move(inner), blockSize](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<BlockLazyTarget<T>>(
        inner(data, opts), static_cast<uint32_t>(data.size()), blockSize);
    // The inner factory has already encoded; calling encode() here would
    // encode a second time.
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

/// The block size the lazy arms are built at: the middle entry of
/// kBlockElementCounts, row-group scale rather than vector scale.
inline constexpr uint32_t kLazyBlockElementCount = 65'536;

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
