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
#include <string>
#include <vector>

#include <glog/logging.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "openzl/cpp/Input.hpp"
#include "openzl/cpp/Output.hpp"
#include "openzl/zl_graphs.h"
#include "openzl/zl_version.h"

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

    size_t compressedSize = cctx.compressOne(
        {compressed_.data(), compressed_.size()}, input);
    compressed_.resize(compressedSize);
  }

  void materializeAll(T* dst, uint32_t n) override {
    openzl::DCtx dctx;
    openzl::Output output = openzl::Output::wrapNumeric(dst, sizeof(T), n);
    dctx.decompressOne(
        output,
        {compressed_.data(), compressed_.size()});
  }

  void materializeRange(uint32_t, uint32_t, T*) override {
    LOG(FATAL) << "OpenZL does not support partial decode";
  }

  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>&,
      T*) override {
    LOG(FATAL) << "OpenZL does not support partial decode";
  }

  size_t payloadSize() const override {
    return compressed_.size();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {{reinterpret_cast<const std::byte*>(compressed_.data()),
             compressed_.size()}};
  }

 private:
  uint32_t count_{0};
  std::vector<char> compressed_;
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
