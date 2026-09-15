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

namespace facebook::nimble::mlidc {

template <typename T>
class OpenZLBenchTarget : public NimbleBenchTargetBase<T> {
 public:
  void encode(const Vector<T>& data, const Encoding::Options&) override {
    count_ = data.size();
    const size_t srcBytes = count_ * sizeof(T);

    openzl::Compressor compressor;
    compressor.selectStartingGraph(
        static_cast<openzl::GraphID>(ZL_StandardGraphID_select_numeric));

    openzl::CCtx cctx;
    cctx.setParameter(
        openzl::CParam::FormatVersion,
        static_cast<int>(ZL_getDefaultEncodingVersion()));
    cctx.refCompressor(compressor);

    openzl::Input input = openzl::Input::refNumeric(data.data(), count_);
    compressed_.resize(openzl::compressBound(srcBytes));

    size_t compressedSize =
        cctx.compressOne({compressed_.data(), compressed_.size()}, input);
    compressed_.resize(compressedSize);
  }

  void materializeAll(T* dst, uint32_t n) override {
    openzl::DCtx dctx;
    openzl::Output output = openzl::Output::wrapNumeric(dst, sizeof(T), n);
    dctx.decompressOne(output, {compressed_.data(), compressed_.size()});
  }

  // OpenZL has no addressable interior, so a partial read is served the only
  // way a block codec can: decompress the whole column, then copy out the rows
  // that were asked for. This is the cost a reader actually pays, and it is
  // the comparison the decode drivers exist to make. It is not a limitation
  // being worked around; reporting it as unsupported would simply leave the
  // comparison unmeasured.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    decompressAll();
    std::copy_n(scratch_.data() + begin, count, dst);
  }

  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) override {
    decompressAll();
    for (const auto& [begin, count] : ranges) {
      std::copy_n(scratch_.data() + begin, count, dst);
      dst += count;
    }
  }

  size_t payloadSize() const override {
    return compressed_.size();
  }

  // Report the codec graph OpenZL chose for this column.
  //
  // The compression driver prints this beside SubIntSplit's section tree when
  // --mlidc_dump_encoding is set, which is what makes the two comparable: the
  // section tree says how SubIntSplit split the word, and this says what the
  // black box did instead. Without it a study can observe only that OpenZL
  // won, never what it did differently.
  //
  // Reflection decompresses the frame to rebuild the graph, so this is only
  // ever called outside a timed region.
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
  // Owns a reflection context so a frame walk cannot leak one on an early
  // return.
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
    openzl::DCtx dctx;
    openzl::Output output =
        openzl::Output::wrapNumeric(scratch_.data(), sizeof(T), count_);
    dctx.decompressOne(output, {compressed_.data(), compressed_.size()});
  }

  uint32_t count_{0};
  std::vector<char> compressed_;
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
  // Every partial read decompresses the whole payload, so drivers bound the
  // iteration count for this entry.
  entry.wholePayloadCodec = true;
  entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<OpenZLBenchTarget<T>>();
    target->encode(data, opts);
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
