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

#include "velox/dwio/nimble/compression/OpenZLCompressor.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "velox/dwio/nimble/common/Exceptions.h"

#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/CParam.hpp"
#include "openzl/cpp/Codecs.hpp"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "openzl/cpp/Input.hpp"
#include "openzl/cpp/Selector.hpp"
#include "openzl/cpp/Type.hpp"

#include "openzl/zl_compressor.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_segmenter.h"
#include "openzl/zl_version.h"

#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/estimate.h"
#include "openzl/shared/numeric_operations.h"

namespace facebook::nimble {
namespace {

// Local parameter ID for element byte width, consumed by the chunk segmenter.
constexpr int kNimbleElementByteWidthParamId = 1;
// Target chunk size: ~16MB.
constexpr size_t kDefaultChunkByteSizeTarget = 16 << 20;

// Multi-chunk segmenter. It splits the serial input into
// ~16MB chunks (aligned to element width) and routes each through the inner
// numeric graph. There is no C++ wrapper for segmenters, so this is registered
// via the C API.
ZL_Report defaultSegmenter(ZL_Segmenter* sctx) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);

  size_t const numInputs = ZL_Segmenter_numInputs(sctx);
  ZL_ERR_IF_NE(numInputs, 1, node_invalid_input);

  const ZL_Input* const input = ZL_Segmenter_getInput(sctx, 0);
  ZL_ASSERT_NN(input);
  size_t const totalBytes = ZL_Input_contentSize(input);

  if (totalBytes == 0) {
    ZL_ERR_IF_ERR(ZL_Segmenter_processChunk(
        sctx, &totalBytes, 1, ZL_GRAPH_STORE, nullptr));
    return ZL_returnSuccess();
  }

  ZL_GraphIDList const customGraphs = ZL_Segmenter_getCustomGraphs(sctx);
  ZL_ASSERT_EQ(customGraphs.nbGraphIDs, 1);
  ZL_GraphID const innerGraph = customGraphs.graphids[0];

  ZL_IntParam const eltWidthParam =
      ZL_Segmenter_getLocalIntParam(sctx, kNimbleElementByteWidthParamId);
  ZL_ERR_IF_EQ(
      eltWidthParam.paramId, ZL_LP_INVALID_PARAMID, node_invalid_input);
  size_t const eltWidth = (size_t)eltWidthParam.paramValue;

  size_t const chunkTarget =
      (kDefaultChunkByteSizeTarget / eltWidth) * eltWidth;

  size_t remaining = totalBytes;
  while (remaining > 0) {
    size_t chunkSize = (remaining > chunkTarget) ? chunkTarget : remaining;
    ZL_ERR_IF_ERR(
        ZL_Segmenter_processChunk(sctx, &chunkSize, 1, innerGraph, nullptr));
    remaining -= chunkSize;
  }
  return ZL_returnSuccess();
}

// Decides if a tokenize transform can and should be used on @p input. It
// calculates an upper bound on the size of tokenization itself (size of
// alphabet + size of indices) and compares it to the original size times a
// threshold multiplier. The multiplier differs for floats and ints.
bool shouldTokenize(
    const openzl::Input& input,
    int compressionLevel,
    bool isFloat) {
  auto const eltWidth = input.eltWidth();
  auto const nbElts = input.numElts();
  auto const src = input.ptr();
  auto const srcSize = eltWidth * nbElts;

  if (compressionLevel == 1 && eltWidth == 8) {
    // Disable tokenization for level 1 and 64-bit integers because it is
    // quite slow.
    return false;
  }

  static constexpr size_t maxAlphabetSizeInBytes =
      64 * 1024; // see T246457068 for details
  const auto maxAlphabetSizeInElts = maxAlphabetSizeInBytes / eltWidth;

  uint64_t const maxEltValue =
      (eltWidth >= 8) ? UINT64_MAX : (1ull << (8 * eltWidth)) - 1;
  uint64_t const maxCardValue =
      std::min(std::min(maxEltValue, (uint64_t)nbElts), maxAlphabetSizeInElts);
  auto const cardinality =
      ZL_estimateCardinality_fixed(src, nbElts, eltWidth, maxCardValue);
  if (cardinality.estimateUpperBound >= maxAlphabetSizeInElts) {
    return false;
  }
  size_t const multiplier = (isFloat || compressionLevel == 1) ? 7 : 9;
  auto const tokenizeEstimatedAlphabetSize =
      cardinality.estimateUpperBound * eltWidth;
  auto const tokenizeEstimatedIndicesSize =
      nbElts * (size_t)ZL_nextPow2(cardinality.estimateUpperBound) / 8;
  auto const tokenizeEstimatedUpperBounds =
      tokenizeEstimatedAlphabetSize + tokenizeEstimatedIndicesSize;
  return tokenizeEstimatedUpperBounds < srcSize * multiplier / 10;
}

// Selects field-lz vs zstd as the backend LZ stage. Single-byte elements go to
// zstd; wider elements use the struct-aware field-lz graph.
class SelectLz : public openzl::Selector {
 public:
  explicit SelectLz(openzl::GraphID fieldLz) : fieldLz_{fieldLz} {}

  openzl::SelectorDescription selectorDescription() const override {
    return {
        .name = std::string{"nimble.select_lz"},
        .inputTypeMask = openzl::TypeMask::Numeric,
        .customGraphs = {fieldLz_},
    };
  }

  openzl::GraphID select(
      openzl::SelectorState& /* state */,
      const openzl::Input& input) const override {
    if (input.eltWidth() == 1) {
      return ZL_GRAPH_ZSTD;
    }
    return fieldLz_;
  }

 private:
  openzl::GraphID fieldLz_;
};

// Selects whether to apply the tokenize stage based on a cardinality estimate.
class SelectTokenize : public openzl::Selector {
 public:
  SelectTokenize(
      openzl::GraphID tokenize,
      openzl::GraphID noTokenize,
      bool isFloat)
      : tokenize_{tokenize}, noTokenize_{noTokenize}, isFloat_{isFloat} {}

  openzl::SelectorDescription selectorDescription() const override {
    return {
        .name = isFloat_ ? "nimble.select_tokenize_f32"
                         : "nimble.select_tokenize_int",
        .inputTypeMask = openzl::TypeMask::Numeric,
        .customGraphs = {tokenize_, noTokenize_},
    };
  }

  openzl::GraphID select(
      openzl::SelectorState& state,
      const openzl::Input& input) const override {
    const int compressionLevel =
        state.getCParam(openzl::CParam::CompressionLevel);
    return shouldTokenize(input, compressionLevel, isFloat_) ? tokenize_
                                                             : noTokenize_;
  }

 private:
  openzl::GraphID tokenize_;
  openzl::GraphID noTokenize_;
  bool isFloat_;
};

// Selects whether to range-pack (subtract min and pack into the smallest
// integer width) based on the actual value range of the input.
class SelectRangePack : public openzl::Selector {
 public:
  SelectRangePack(openzl::GraphID rangePack, openzl::GraphID noRangePack)
      : rangePack_{rangePack}, noRangePack_{noRangePack} {}

  openzl::SelectorDescription selectorDescription() const override {
    return {
        .name = std::string{"nimble.select_range_pack"},
        .inputTypeMask = openzl::TypeMask::Numeric,
        .customGraphs = {rangePack_, noRangePack_},
    };
  }

  openzl::GraphID select(
      openzl::SelectorState& /* state */,
      const openzl::Input& input) const override {
    const auto eltWidth = input.eltWidth();
    const auto nbElts = input.numElts();
    const auto src = input.ptr();
    const ZL_ElementRange range =
        ZL_computeUnsignedRange(src, nbElts, eltWidth);
    const auto rangeSize = range.max - range.min;
    if (NUMOP_numericWidthForValue(rangeSize) < eltWidth) {
      return rangePack_;
    }
    return noRangePack_;
  }

 private:
  openzl::GraphID rangePack_;
  openzl::GraphID noRangePack_;
};

// Registers the chunk segmenter on the raw compressor and wraps @p innerGraph
// with it, threading the element byte width through as a local parameter. This
// is the only place the C API is used directly, since segmenters have no C++
// wrapper.
openzl::GraphID wrapWithDefaultSegmenter(
    ZL_Compressor* cgraph,
    openzl::GraphID innerGraph,
    size_t elementByteWidth) {
  ZL_Type inputType = ZL_Type_serial;
  ZL_SegmenterDesc desc = {
      .name = "!nimble.default_segmenter",
      .segmenterFn = defaultSegmenter,
      .inputTypeMasks = &inputType,
      .numInputs = 1,
      .lastInputIsVariable = false,
      .customGraphs = nullptr,
      .numCustomGraphs = 0,
      .localParams = {},
  };
  ZL_GraphID const segmenterBase =
      ZL_Compressor_registerSegmenter(cgraph, &desc);

  ZL_IntParam intParams[] = {{
      .paramId = kNimbleElementByteWidthParamId,
      .paramValue = static_cast<int>(elementByteWidth),
  }};
  ZL_LocalParams segParams = {
      .intParams = {.intParams = intParams, .nbIntParams = 1},
  };
  ZL_ParameterizedGraphDesc const segGraphDesc = {
      .graph = segmenterBase,
      .customGraphs = &innerGraph,
      .nbCustomGraphs = 1,
      .localParams = &segParams,
  };
  return ZL_Compressor_registerParameterizedGraph(cgraph, &segGraphDesc);
}

// Builds the default numeric graph:
//   serial -> interpret-as-LE(width) -> range-pack? -> tokenize?+delta -> lz
// Wrapped by the multi-chunk segmenter.
// This decision-making is a default "should-be-good-enough" pipeline that can
// be modified or replaced to fit to your specific use case.
openzl::GraphID makeNumericGraph(
    openzl::Compressor& compressor,
    bool isFloat,
    size_t elementBitWidth,
    int formatVersion) {
  const size_t elementByteWidth = elementBitWidth / 8;

  const openzl::GraphID fieldLz = openzl::graphs::FieldLz{}(compressor);
  const openzl::GraphID lz = openzl::Selector::registerSelector(
      compressor, std::make_shared<SelectLz>(fieldLz));

  const openzl::GraphID deltaLz = openzl::nodes::DeltaInt{}(compressor, lz);
  const openzl::GraphID tokenize =
      openzl::nodes::TokenizeNumeric{/* sort */ true}(compressor, deltaLz, lz);

  const openzl::GraphID tokenizeOrNot = openzl::Selector::registerSelector(
      compressor, std::make_shared<SelectTokenize>(tokenize, lz, isFloat));

  const openzl::GraphID range =
      openzl::nodes::RangePack{}(compressor, tokenizeOrNot);
  const openzl::GraphID rangeOrNot = openzl::Selector::registerSelector(
      compressor, std::make_shared<SelectRangePack>(range, tokenizeOrNot));

  openzl::nodes::ConvertSerialToNumLE toNumeric{
      static_cast<int>(elementByteWidth)};
  const openzl::GraphID innerGraph = toNumeric(compressor, rangeOrNot);
  return wrapWithDefaultSegmenter(
      compressor.get(), innerGraph, elementByteWidth);
}

// Maps the nimble DataType to the appropriate numeric graph. Types without
// a dedicated numeric pipeline fall back to a plain zstd graph (still a valid
// OpenZL frame).
openzl::GraphID createGraph(
    openzl::Compressor& compressor,
    DataType dataType,
    int formatVersion) {
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
      return makeNumericGraph(
          compressor, /* isFloat */ false, 8, formatVersion);
    case DataType::Int16:
    case DataType::Uint16:
      return makeNumericGraph(
          compressor, /* isFloat */ false, 16, formatVersion);
    case DataType::Int32:
    case DataType::Uint32:
      return makeNumericGraph(
          compressor, /* isFloat */ false, 32, formatVersion);
    case DataType::Int64:
    case DataType::Uint64:
      return makeNumericGraph(
          compressor, /* isFloat */ false, 64, formatVersion);
    case DataType::Float:
      return makeNumericGraph(
          compressor, /* isFloat */ true, 32, formatVersion);
    case DataType::Double:
    case DataType::Bool:
    case DataType::String:
    case DataType::Undefined:
    default:
      return ZL_GRAPH_ZSTD;
  }
}

} // namespace

CompressionResult OpenZLCompressor::compress(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    DataType dataType,
    int /* bitWidth */,
    const CompressionPolicy& compressionPolicy) {
  const auto parameters = compressionPolicy.config().parameters.openzl;

  openzl::Compressor compressor;
  compressor.selectStartingGraph(
      createGraph(compressor, dataType, parameters.formatVersion));

  openzl::CCtx cctx;
  cctx.setParameter(
      openzl::CParam::CompressionLevel, parameters.compressionLevel);
  cctx.setParameter(
      openzl::CParam::DecompressionLevel, parameters.decompressionLevel);
  cctx.setParameter(openzl::CParam::FormatVersion, parameters.formatVersion);
  cctx.refCompressor(compressor);

  Vector<char> buffer{&pool, openzl::compressBound(data.size())};
  const size_t compressedSize =
      cctx.compressSerial({buffer.data(), buffer.size()}, data);

  if (!compressionPolicy.shouldAccept(
          CompressionType::OpenZL, data.size(), compressedSize)) {
    return {
        .compressionType = CompressionType::Uncompressed,
        .buffer = std::nullopt,
    };
  }

  buffer.resize(compressedSize);
  return {
      .compressionType = CompressionType::OpenZL,
      .buffer = std::move(buffer),
  };
}

velox::BufferPtr OpenZLCompressor::uncompress(
    velox::memory::MemoryPool& pool,
    const CompressionType /* compressionType */,
    const DataType /* dataType */,
    std::string_view data,
    velox::BufferPool* bufferPool) {
  openzl::DCtx dctx;
  const size_t uncompressedSize =
      dctx.unwrap(ZL_getDecompressedSize(data.data(), data.size()));
  auto buffer = allocateBuffer(pool, bufferPool, uncompressedSize);
  const size_t written =
      dctx.decompressSerial({buffer->asMutable<char>(), buffer->size()}, data);
  NIMBLE_CHECK(
      written == uncompressedSize,
      "Decompressed size mismatch: expected {}, got {}",
      uncompressedSize,
      written);
  return buffer;
}

std::optional<size_t> OpenZLCompressor::uncompressedSize(
    std::string_view data) const {
  const ZL_Report report = ZL_getDecompressedSize(data.data(), data.size());
  if (ZL_isError(report)) {
    return std::nullopt;
  }
  return ZL_validResult(report);
}

CompressionType OpenZLCompressor::compressionType() {
  return CompressionType::OpenZL;
}

} // namespace facebook::nimble
