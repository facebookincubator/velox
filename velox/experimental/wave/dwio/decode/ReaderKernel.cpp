/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/experimental/wave/dwio/decode/ReaderKernel.h"

#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <fstream>

#include "velox/common/base/Exceptions.h"
#include "velox/experimental/wave/dwio/decode/DecodeStep.h"

DEFINE_bool(
    wave_print_reader_kernels,
    false,
    "Write the source of each new compiled reader kernel to /tmp");

namespace facebook::velox::wave {

namespace {

const char* stepName(DecodeStep step) {
  switch (step) { // NOLINT(clang-diagnostic-switch-enum)
    case DecodeStep::kSelective32:
      return "kSelective32";
    case DecodeStep::kSelective64:
      return "kSelective64";
    case DecodeStep::kSelective32Chunked:
      return "kSelective32Chunked";
    case DecodeStep::kSelective64Chunked:
      return "kSelective64Chunked";
    case DecodeStep::kCompact64:
      return "kCompact64";
    case DecodeStep::kTrivial:
      return "kTrivial";
    case DecodeStep::kDictionaryOnBitpack:
      return "kDictionaryOnBitpack";
    case DecodeStep::kSparseBool:
      return "kSparseBool";
    case DecodeStep::kMainlyConstant:
      return "kMainlyConstant";
    case DecodeStep::kVarint:
      return "kVarint";
    case DecodeStep::kRleTotalLength:
      return "kRleTotalLength";
    case DecodeStep::kRle:
      return "kRle";
    case DecodeStep::kMakeScatterIndices:
      return "kMakeScatterIndices";
    case DecodeStep::kRowCountNoFilter:
      return "kRowCountNoFilter";
    case DecodeStep::kCountBits:
      return "kCountBits";
    case DecodeStep::kNimble:
      return "kNimble";
    case DecodeStep::kNimblePrepare:
      return "kNimblePrepare";
    case DecodeStep::kNested:
      return "kNested";
    case DecodeStep::kExpandLengths:
      return "kExpandLengths";
    default:
      VELOX_FAIL(
          "DecodeStep has no compiled reader implementation: {}",
          static_cast<int32_t>(step));
  }
}

const char* filterName(WaveFilterKind kind) {
  switch (kind) { // NOLINT(clang-diagnostic-switch-enum)
    case WaveFilterKind::kAlwaysTrue:
      return "kAlwaysTrue";
    case WaveFilterKind::kBigintRange:
      return "kBigintRange";
    case WaveFilterKind::kDictFilter:
      return "kDictFilter";
    default:
      VELOX_FAIL(
          "Filter kind has no compiled reader implementation: {}",
          static_cast<int32_t>(kind));
  }
}

const char* nullModeName(NullMode mode) {
  switch (mode) {
    case NullMode::kDenseNonNull:
      return "kDenseNonNull";
    case NullMode::kSparseNonNull:
      return "kSparseNonNull";
    case NullMode::kDenseNullable:
      return "kDenseNullable";
    case NullMode::kSparseNullable:
      return "kSparseNullable";
  }
  VELOX_UNREACHABLE();
}

const char* dictModeName(DictMode mode) {
  switch (mode) {
    case DictMode::kNone:
      return "kNone";
    case DictMode::kDict:
      return "kDict";
    case DictMode::kDictFilter:
      return "kDictFilter";
    case DictMode::kRecordFilter:
      return "kRecordFilter";
  }
  VELOX_UNREACHABLE();
}

// Maps a type to the integer type of the same width. The decoders move bits,
// so REAL decodes as INTEGER and DOUBLE as BIGINT.
WaveTypeKind integerKind(WaveTypeKind kind) {
  switch (kind) { // NOLINT(clang-diagnostic-switch-enum)
    case WaveTypeKind::TINYINT:
    case WaveTypeKind::SMALLINT:
    case WaveTypeKind::INTEGER:
    case WaveTypeKind::BIGINT:
      return kind;
    case WaveTypeKind::REAL:
      return WaveTypeKind::INTEGER;
    case WaveTypeKind::DOUBLE:
      return WaveTypeKind::BIGINT;
    default:
      VELOX_FAIL(
          "Type has no compiled reader implementation: {}",
          static_cast<int32_t>(kind));
  }
}

const char* typeName(WaveTypeKind kind, bool isSigned) {
  switch (kind) { // NOLINT(clang-diagnostic-switch-enum)
    case WaveTypeKind::TINYINT:
      return isSigned ? "int8_t" : "uint8_t";
    case WaveTypeKind::SMALLINT:
      return isSigned ? "int16_t" : "uint16_t";
    case WaveTypeKind::INTEGER:
      return isSigned ? "int32_t" : "uint32_t";
    case WaveTypeKind::BIGINT:
      return isSigned ? "int64_t" : "uint64_t";
    case WaveTypeKind::VARCHAR:
      return "StringView";
    default:
      VELOX_UNREACHABLE();
  }
}

bool isNullable(NullMode mode) {
  return mode == NullMode::kDenseNullable || mode == NullMode::kSparseNullable;
}

struct AccessorRegistry {
  std::vector<std::string> accessors;
  folly::F14FastMap<std::string, int32_t> ids;
};

folly::Synchronized<AccessorRegistry>& accessorRegistry() {
  static folly::Synchronized<AccessorRegistry> registry;
  return registry;
}

void setNimbleSignature(const GpuDecode& op, DecodeSignature& signature) {
  VELOX_CHECK(
      op.nullMode == NullMode::kDenseNonNull ||
          op.nullMode == NullMode::kSparseNonNull,
      "Nimble decode does not support nulls");
  VELOX_CHECK(
      op.filterKind == WaveFilterKind::kAlwaysTrue ||
          op.filterKind == WaveFilterKind::kBigintRange,
      "Filter kind has no compiled Nimble implementation: {}",
      static_cast<int32_t>(op.filterKind));
  signature.encoding = op.encoding;
  if (op.dataType == WaveTypeKind::VARCHAR ||
      op.dataType == WaveTypeKind::VARBINARY) {
    VELOX_CHECK(
        op.filterKind == WaveFilterKind::kAlwaysTrue,
        "Filters on strings have no compiled Nimble implementation");
    signature.dataType = WaveTypeKind::VARCHAR;
  } else {
    signature.dataType = op.step == DecodeStep::kSelective64
        ? WaveTypeKind::BIGINT
        : WaveTypeKind::INTEGER;
  }
  signature.nullMode = op.nullMode;
  signature.filterKind = op.filterKind;
  if (signature.filterKind != WaveFilterKind::kAlwaysTrue) {
    signature.hasResult = op.result != nullptr;
  }
  signature.variant = op.column().accessorsId;
}

// Chooses the index type and normalizes the dict mode. Only kRecordFilter
// changes the instantiation of a selective op beyond the filter and index
// type.
void setSelectiveSignature(const GpuDecode& op, DecodeSignature& signature) {
  if (op.encoding == DecodeStep::kNimble) {
    setNimbleSignature(op, signature);
    return;
  }
  VELOX_CHECK(
      op.encoding == DecodeStep::kDictionaryOnBitpack,
      "Selective decode only supports kDictionaryOnBitpack encoding: {}",
      static_cast<int32_t>(op.encoding));
  const bool is64 = op.step == DecodeStep::kSelective64;
  signature.encoding = op.encoding;
  signature.dataType = is64 ? WaveTypeKind::BIGINT : WaveTypeKind::INTEGER;
  signature.indexType = signature.dataType;
  signature.nullMode = op.nullMode;
  if (op.dictMode == DictMode::kDictFilter) {
    signature.filterKind = WaveFilterKind::kDictFilter;
    signature.indexType = WaveTypeKind::INTEGER;
  } else if (op.filterKind == WaveFilterKind::kAlwaysTrue) {
    if (is64 && op.dictMode == DictMode::kDict) {
      signature.indexType = WaveTypeKind::INTEGER;
    }
  } else if (op.filterKind == WaveFilterKind::kBigintRange) {
    signature.filterKind = WaveFilterKind::kBigintRange;
    if (op.dictMode == DictMode::kRecordFilter) {
      signature.dictMode = DictMode::kRecordFilter;
    }
  } else {
    VELOX_FAIL(
        "Filter kind has no compiled reader implementation: {}",
        static_cast<int32_t>(op.filterKind));
  }
  // Without a filter all values are written whether or not the op has a
  // result.
  if (signature.filterKind != WaveFilterKind::kAlwaysTrue) {
    signature.hasResult = op.result != nullptr;
  }
}

} // namespace

int32_t accessorListId(const std::string& accessors) {
  return accessorRegistry().withWLock([&](auto& registry) {
    auto [it, isNew] =
        registry.ids.emplace(accessors, registry.accessors.size());
    if (isNew) {
      registry.accessors.push_back(accessors);
    }
    return it->second;
  });
}

std::string accessorList(int32_t id) {
  return accessorRegistry().withRLock([&](const auto& registry) {
    VELOX_CHECK_LT(id, registry.accessors.size());
    return registry.accessors[id];
  });
}

DecodeSignature decodeSignature(const GpuDecode& op) {
  DecodeSignature signature;
  signature.step = op.step;
  switch (op.step) { // NOLINT(clang-diagnostic-switch-enum)
    case DecodeStep::kSelective32:
    case DecodeStep::kSelective64:
      setSelectiveSignature(op, signature);
      break;
    case DecodeStep::kSelective32Chunked:
    case DecodeStep::kSelective64Chunked:
      signature.nullMode = isNullable(op.nullMode) ? NullMode::kDenseNullable
                                                   : NullMode::kDenseNonNull;
      break;
    case DecodeStep::kTrivial:
      signature.dataType = integerKind(op.data.trivial.dataType);
      break;
    case DecodeStep::kDictionaryOnBitpack:
      VELOX_CHECK(
          op.data.dictionaryOnBitpack.dataType != WaveTypeKind::REAL &&
              op.data.dictionaryOnBitpack.dataType != WaveTypeKind::DOUBLE,
          "kDictionaryOnBitpack does not support floating point");
      signature.dataType = integerKind(op.data.dictionaryOnBitpack.dataType);
      break;
    case DecodeStep::kMainlyConstant:
      signature.dataType = integerKind(op.data.mainlyConstant.dataType);
      break;
    case DecodeStep::kRle:
      signature.dataType = integerKind(op.data.rle.valueType);
      break;
    case DecodeStep::kVarint:
      VELOX_CHECK(
          op.data.varint.resultType == WaveTypeKind::INTEGER ||
              op.data.varint.resultType == WaveTypeKind::BIGINT,
          "kVarint only supports INTEGER and BIGINT");
      signature.dataType = op.data.varint.resultType;
      break;
    case DecodeStep::kCountBits: {
      auto stride = op.data.countBits.resultStride;
      VELOX_CHECK(
          stride == 256 || stride == 512 || stride == 1024 || stride == 2048,
          "kCountBits stride must be 256, 512, 1024 or 2048: {}",
          stride);
      signature.variant = stride;
      break;
    }
    case DecodeStep::kCompact64:
      // Compacts values of the column type, also 32 bit ones.
      signature.dataType = integerKind(op.dataType);
      VELOX_CHECK(
          signature.dataType == WaveTypeKind::INTEGER ||
              signature.dataType == WaveTypeKind::BIGINT,
          "kCompact64 only supports 4 and 8 byte types");
      break;
    case DecodeStep::kNimblePrepare:
    case DecodeStep::kNested:
    case DecodeStep::kExpandLengths:
    case DecodeStep::kSparseBool:
    case DecodeStep::kRleTotalLength:
    case DecodeStep::kMakeScatterIndices:
    case DecodeStep::kRowCountNoFilter:
      break;
    default:
      VELOX_FAIL(
          "DecodeStep has no compiled reader implementation: {}",
          static_cast<int32_t>(op.step));
  }
  return signature;
}

std::string DecodeSignature::toString() const {
  if (encoding == DecodeStep::kNimble) {
    // The accessor id is process local. The text identifies the kernel.
    DecodeSignature copy = *this;
    copy.variant = 0;
    copy.encoding = DecodeStep::kUnsupported;
    return fmt::format("{}.N[{}]", copy.toString(), accessorList(variant));
  }
  return fmt::format(
      "{}.{}.{}.{}.{}.{}.{}.{}.{}",
      static_cast<int32_t>(step),
      static_cast<int32_t>(encoding),
      static_cast<int32_t>(dataType),
      static_cast<int32_t>(indexType),
      static_cast<int32_t>(filterKind),
      static_cast<int32_t>(nullMode),
      static_cast<int32_t>(dictMode),
      hasResult ? 1 : 0,
      variant);
}

std::string DecodeSignature::call(int32_t blockSize) const {
  switch (step) { // NOLINT(clang-diagnostic-switch-enum)
    case DecodeStep::kSelective32:
    case DecodeStep::kSelective64:
      if (encoding == DecodeStep::kNimble) {
        return fmt::format(
            "{{ using namespace detail; "
            "decodeColumn<NimbleFormat, {}, {}, WaveFilterKind::{}, {}, {}, "
            "{}>(&op); }}",
            typeName(dataType, true),
            blockSize,
            filterName(filterKind),
            hasResult ? "true" : "false",
            nullMode == NullMode::kSparseNonNull ? "true" : "false",
            accessorList(variant));
      }
      return fmt::format(
          "detail::decodeSelective<{}, {}, DecodeStep::{}, WaveFilterKind::{}, "
          "{}, {}, NullMode::{}, DictMode::{}>(&op);",
          typeName(dataType, true),
          blockSize,
          stepName(encoding),
          filterName(filterKind),
          hasResult ? "true" : "false",
          typeName(indexType, true),
          nullModeName(nullMode),
          dictModeName(dictMode));
    case DecodeStep::kSelective32Chunked:
    case DecodeStep::kSelective64Chunked:
      return fmt::format(
          "detail::selectiveFilterChunked<{}, {}, {}>(&op);",
          step == DecodeStep::kSelective32Chunked ? "int32_t" : "int64_t",
          blockSize,
          isNullable(nullMode) ? "true" : "false");
    case DecodeStep::kCompact64:
      return fmt::format(
          "detail::compactValues<{}, {}>(&op);",
          typeName(dataType, true),
          blockSize);
    case DecodeStep::kCountBits:
      return fmt::format("detail::countBits<{}, {}>(op);", blockSize, variant);
    case DecodeStep::kTrivial:
      return fmt::format(
          "detail::decodeTrivial<{}>(op.data.trivial);",
          typeName(dataType, false));
    case DecodeStep::kDictionaryOnBitpack:
      return fmt::format(
          "detail::decodeDictionaryOnBitpack<{}>(op.data.dictionaryOnBitpack);",
          typeName(dataType, true));
    case DecodeStep::kSparseBool:
      return "detail::decodeSparseBool(op.data.sparseBool);";
    case DecodeStep::kNimblePrepare:
      return fmt::format("detail::prepareNimbleNode<{}>(&op);", blockSize);
    case DecodeStep::kNested:
      return fmt::format("detail::nestedRows<{}>(&op);", blockSize);
    case DecodeStep::kExpandLengths:
      return fmt::format("detail::expandLengths<{}>(&op);", blockSize);
    case DecodeStep::kMainlyConstant:
      return fmt::format(
          "detail::decodeMainlyConstant<{}, {}>(op.data.mainlyConstant);",
          blockSize,
          typeName(dataType, false));
    case DecodeStep::kVarint:
      return fmt::format(
          "detail::decodeVarintOp<{}, {}>(op.data.varint);",
          blockSize,
          typeName(dataType, false));
    case DecodeStep::kRleTotalLength:
      return fmt::format(
          "detail::rleTotalLength<{}>(op.data.rleTotalLength);", blockSize);
    case DecodeStep::kRle:
      return fmt::format(
          "detail::decodeRle<{}, {}>(op.data.rle);",
          blockSize,
          typeName(dataType, false));
    case DecodeStep::kMakeScatterIndices:
      return fmt::format(
          "detail::makeScatterIndices<{}>(op.data.makeScatterIndices);",
          blockSize);
    case DecodeStep::kRowCountNoFilter:
      return fmt::format(
          "detail::setRowCountNoFilter<{}>(op.data.rowCountNoFilter);",
          blockSize);
    default:
      VELOX_UNREACHABLE();
  }
}

std::string ReaderKernelKey::toString() const {
  std::string result = fmt::format("wave_reader:{}", blockSize);
  for (const auto& signature : signatures) {
    result += ":";
    result += signature.toString();
  }
  return result;
}

int32_t ReaderKernelKey::opcode(const DecodeSignature& signature) const {
  auto it = std::lower_bound(signatures.begin(), signatures.end(), signature);
  VELOX_CHECK(it != signatures.end() && *it == signature);
  return static_cast<int32_t>(it - signatures.begin());
}

int32_t ReaderKernelKey::sharedMemorySize() const {
  int32_t size{0};
  for (const auto& signature : signatures) {
    size = std::max(size, decodeSharedMemorySize(signature.step, blockSize));
  }
  // Allow aligning at 16.
  return size > 0 ? size + 15 : 0;
}

ReaderKernelKey makeReaderKernelKey(
    const std::vector<GpuDecode*>& ops,
    int32_t blockSize) {
  ReaderKernelKey key;
  key.blockSize = blockSize;
  std::vector<DecodeSignature> opSignatures;
  opSignatures.reserve(ops.size());
  for (const auto* op : ops) {
    opSignatures.push_back(decodeSignature(*op));
  }
  key.signatures = opSignatures;
  std::sort(key.signatures.begin(), key.signatures.end());
  key.signatures.erase(
      std::unique(key.signatures.begin(), key.signatures.end()),
      key.signatures.end());
  VELOX_CHECK_LE(
      key.signatures.size(),
      256,
      "Too many distinct decode ops in one reader kernel");
  for (auto i = 0; i < ops.size(); ++i) {
    ops[i]->opcode = key.opcode(opSignatures[i]);
  }
  return key;
}

namespace {

std::string readerOpText(const ReaderKernelKey& key) {
  std::string text =
      "__device__ __forceinline__ void readerOp(GpuDecode& op) {\n";
  if (key.signatures.size() == 1) {
    text += fmt::format("  {}\n}}\n\n", key.signatures[0].call(key.blockSize));
    return text;
  }
  text += "  switch (op.opcode) {\n";
  for (auto i = 0; i < key.signatures.size(); ++i) {
    text += fmt::format(
        "    case {}:\n      {}\n      break;\n",
        i,
        key.signatures[i].call(key.blockSize));
  }
  text += "  }\n}\n\n";
  return text;
}

// waveReader runs the ops of the TB's program from GpuDecodeParams.
// waveReaderSingle runs one op per TB, copied to shared memory with space for
// 'temp' after it.
constexpr const char* kReaderEntries = R"(
__global__ void __launch_bounds__({0}) waveReader(GpuDecodeParams inlineParams) {{
  __shared__ GpuDecodeParams* params;
  __shared__ int32_t programStart;
  __shared__ int32_t programEnd;
  __shared__ GpuDecode* ops;
  if (threadIdx.x == 0) {{
    params = inlineParams.external ? inlineParams.external : &inlineParams;
    programStart = blockIdx.x == 0 ? 0 : params->ends[blockIdx.x - 1];
    programEnd = params->ends[blockIdx.x];
    ops = reinterpret_cast<GpuDecode*>(&params->ends[0] + roundUp(gridDim.x, 4));
  }}
  __syncthreads();
  for (auto i = programStart; i < programEnd; ++i) {{
    readerOp(ops[i]);
  }}
  __syncthreads();
}}

__global__ void __launch_bounds__({0}) waveReaderSingle(GpuDecode* plan) {{
  constexpr int32_t kOpSize = 1 + sizeof(GpuDecode) / 8;
  __shared__ int64_t shared[kOpSize + 8];
  if (threadIdx.x == 0) {{
    auto op = reinterpret_cast<int64_t*>(&plan[blockIdx.x]);
    for (auto i = 0; i < kOpSize; ++i) {{
      shared[i] = op[i];
    }}
    reinterpret_cast<GpuDecode*>(&shared)->temp =
        reinterpret_cast<int32_t*>(&shared[kOpSize]);
  }}
  __syncthreads();
  readerOp(*reinterpret_cast<GpuDecode*>(&shared));
  __syncthreads();
}}
)";

folly::Synchronized<folly::F14FastSet<std::string>>& requestedKeys() {
  static folly::Synchronized<folly::F14FastSet<std::string>> keys;
  return keys;
}

} // namespace

KernelSpec readerKernelSpec(const ReaderKernelKey& key) {
  KernelSpec spec;
  spec.code =
      "#include \"velox/experimental/wave/dwio/decode/NimbleOps.cuh\"\n"
      "\nnamespace facebook::velox::wave {\n\n" +
      readerOpText(key) + fmt::format(kReaderEntries, key.blockSize) +
      "\n} // namespace facebook::velox::wave\n";
  spec.entryPoints = {
      "facebook::velox::wave::waveReader",
      "facebook::velox::wave::waveReaderSingle"};
  spec.filePath = fmt::format(
      "/tmp/wave_reader_{}.cu", std::hash<std::string>{}(key.toString()));
  return spec;
}

ReaderKernel getReaderKernel(const ReaderKernelKey& key) {
  auto keyString = key.toString();
  const bool isNew = requestedKeys().wlock()->insert(keyString).second;
  if (isNew && FLAGS_wave_print_reader_kernels) {
    auto spec = readerKernelSpec(key);
    std::ofstream out(spec.filePath, std::ios_base::out | std::ios_base::trunc);
    out << "// " << keyString << "\n" << spec.code;
  }
  return {
      CompiledKernel::getKernel(
          keyString, [key]() { return readerKernelSpec(key); }),
      isNew};
}

} // namespace facebook::velox::wave
