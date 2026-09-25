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

#include <cstdint>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "openzl/cpp/Input.hpp"
#include "openzl/cpp/Output.hpp"
#include "openzl/zl_graphs.h"
#include "openzl/zl_reflection.h"
#include "openzl/zl_version.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BlockCodecTarget.h"

namespace facebook::nimble::mlidc {

/// Compresses `count` elements with OpenZL's select_numeric graph, appending
/// the frame to `out`. Returns the frame's size. Shared by the whole-column
/// arm and the block arms so openzl/block-K differs from openzl/auto only
/// in block size, not in graph setup.
template <typename T>
size_t openzlCompressNumeric(const T* src, size_t count, std::string& out) {
  openzl::Compressor compressor;
  compressor.selectStartingGraph(
      static_cast<openzl::GraphID>(ZL_StandardGraphID_select_numeric));

  openzl::CCtx cctx;
  cctx.setParameter(
      openzl::CParam::FormatVersion,
      static_cast<int>(ZL_getDefaultEncodingVersion()));
  cctx.refCompressor(compressor);

  openzl::Input input = openzl::Input::refNumeric(src, count);
  const size_t offset = out.size();
  out.resize(offset + openzl::compressBound(count * sizeof(T)));
  const size_t compressedSize =
      cctx.compressOne({out.data() + offset, out.size() - offset}, input);
  out.resize(offset + compressedSize);
  return compressedSize;
}

/// Decompresses one frame produced by openzlCompressNumeric into `dst`, which
/// has room for `count` elements.
template <typename T>
void openzlDecompressNumeric(std::string_view frame, size_t count, T* dst) {
  openzl::DCtx dctx;
  openzl::Output output = openzl::Output::wrapNumeric(dst, sizeof(T), count);
  dctx.decompressOne(output, {frame.data(), frame.size()});
}

/// BlockCodec applying the same OpenZL graph one block at a time.
template <typename T>
class OpenZLBlockCodec : public BlockCodec<T> {
 public:
  bool compressBlock(const T* src, uint32_t count, std::string& out) override {
    openzlCompressNumeric<T>(src, count, out);
    // OpenZL always produces a frame, so unlike Zstd there is no declined
    // block and nothing is ever stored raw.
    return true;
  }

  void decompressBlock(std::string_view block, uint32_t count, T* dst)
      override {
    openzlDecompressNumeric<T>(block, count, dst);
  }
};

/// The OpenZL block arms, the addressable counterpart to openzl/auto.
template <typename T>
std::vector<EncoderEntry<T>> buildOpenZLBlockEncoders() {
  std::vector<EncoderEntry<T>> entries;
  entries.reserve(kBlockElementCounts.size());
  for (const uint32_t blockSize : kBlockElementCounts) {
    entries.push_back(
        makeBlockCodecEntry<T>(
            "openzl",
            "OpenZL",
            blockSize,
            []() -> std::unique_ptr<BlockCodec<T>> {
              return std::make_unique<OpenZLBlockCodec<T>>();
            }));
  }
  return entries;
}

template <typename T>
class OpenZLBenchTarget : public NimbleBenchTargetBase<T> {
 public:
  void encode(const Vector<T>& data, const Encoding::Options&) override {
    count_ = data.size();
    compressed_.clear();
    openzlCompressNumeric<T>(data.data(), count_, compressed_);
  }

  void materializeAll(T* dst, uint32_t n) override {
    openzlDecompressNumeric<T>(
        std::string_view{compressed_.data(), compressed_.size()}, n, dst);
  }

  // OpenZL has no addressable interior, so a partial read decompresses the
  // whole column, then copies out the rows that were asked for. This is the
  // cost a reader actually pays, and the comparison the decode drivers exist
  // to make.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    decompressAll();
    std::copy_n(scratch_.data() + begin, count, dst);
  }

  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    decompressAll();
    for (const auto& range : ranges) {
      std::copy_n(scratch_.data() + range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
  }

  size_t payloadSize() const override {
    return compressed_.size();
  }

  // The frame plus the scratch buffer a partial read decompresses into,
  // which holds a whole decoded column kept between reads.
  size_t residentBytes() const override {
    return compressed_.size() + scratch_.capacity() * sizeof(T);
  }

  // No addressable interior at all, so every read decompresses the frame.
  ReadPath readPath() const override {
    return ReadPath::kWholePayload;
  }

  // Reports the codec graph OpenZL chose for this column, alongside
  // SubIntSplit's section tree, so a study can see what the black box did
  // rather than only that it won. Reflection decompresses the frame to
  // rebuild the graph, so this is only ever called outside a timed region.
  std::string describe() override {
    if (compressed_.empty()) {
      return {};
    }

    ReflectionContext reflection;
    if (!reflection.valid()) {
      return {};
    }
    const ZL_Report report = ZL_ReflectionCtx_setCompressedFrame(
        reflection.get(), compressed_.data(), compressed_.size());
    if (ZL_isError(report)) {
      // Stay silent rather than print a half-built graph.
      return {};
    }

    ZL_ReflectionCtx* rctx = reflection.get();
    const size_t numCodecs = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);

    std::ostringstream out;
    out << "OpenZLGraph codecs=" << numCodecs
        << " frameHeaderBytes=" << ZL_ReflectionCtx_getFrameHeaderSize(rctx)
        << " storedOutputs="
        << ZL_ReflectionCtx_getNumStoredOutputs_lastChunk(rctx) << "\n";

    for (size_t i = 0; i < numCodecs; ++i) {
      const ZL_CodecInfo* codec = ZL_ReflectionCtx_getCodec_lastChunk(rctx, i);
      if (codec == nullptr) {
        continue;
      }
      const char* name = ZL_CodecInfo_getName(codec);
      out << "  [" << i << "] " << (name != nullptr ? name : "<unnamed>")
          << (ZL_CodecInfo_isStandardCodec(codec) ? " standard" : " custom")
          << " id=" << ZL_CodecInfo_getCodecID(codec);

      const size_t numOutputs = ZL_CodecInfo_getNumOutputs(codec);
      size_t outputBytes = 0;
      for (size_t output = 0; output < numOutputs; ++output) {
        const ZL_DataInfo* stream = ZL_CodecInfo_getOutput(codec, output);
        if (stream != nullptr) {
          outputBytes += ZL_DataInfo_getContentSize(stream);
        }
      }
      out << " outputs=" << numOutputs << " outputBytes=" << outputBytes
          << "\n";
    }
    return out.str();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(compressed_.data()),
         compressed_.size()}};
  }

 private:
  // Owns a reflection context so a frame walk cannot leak on early return.
  class ReflectionContext {
   public:
    ReflectionContext() : rctx_(ZL_ReflectionCtx_create()) {}

    ~ReflectionContext() {
      if (rctx_ != nullptr) {
        ZL_ReflectionCtx_free(rctx_);
      }
    }

    ReflectionContext(const ReflectionContext&) = delete;
    ReflectionContext& operator=(const ReflectionContext&) = delete;

    bool valid() const {
      return rctx_ != nullptr;
    }

    ZL_ReflectionCtx* get() const {
      return rctx_;
    }

   private:
    ZL_ReflectionCtx* rctx_;
  };

  // Charged on every partial read, never cached across calls: a reader holding
  // a compressed block pays this each time it needs rows.
  void decompressAll() {
    scratch_.resize(count_);
    openzlDecompressNumeric<T>(
        std::string_view{compressed_.data(), compressed_.size()},
        count_,
        scratch_.data());
  }

  uint32_t count_{0};
  std::string compressed_;
  std::vector<T> scratch_;
};

template <typename T>
EncoderEntry<T> buildOpenZLEncoder() {
  EncoderEntry<T> entry;
  entry.name = "openzl/auto";
  entry.family = "OpenZL";
  entry.variant = "select_numeric";
  entry.isSequential = true;
  entry.fastSkip = false;
  entry.randomAccess = false;
  entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<OpenZLBenchTarget<T>>();
    target->encode(data, opts);
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
