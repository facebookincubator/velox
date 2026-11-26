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

#include "velox/experimental/wave/exec/tests/utils/TestFormatReader.h"
#include "velox/experimental/wave/dwio/StructColumnReader.h"

DECLARE_int32(wave_reader_rows_per_tb);

namespace facebook::velox::wave::test {

using common::Subfield;

std::unique_ptr<FormatData> TestFormatParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const velox::common::ScanSpec& scanSpec,
    OperandId operand) {
  auto* column = type->id() == 0 ? nullptr : stripe_->findColumn(*type);
  return std::make_unique<TestFormatData>(
      operand, stripe_->columns[0]->numValues, column);
}

BufferId TestFormatData::stageNulls(
    ResultStaging& deviceStaging,
    SplitStaging& splitStaging) {
  if (!column_->nulls) {
    nullsStaged_ = true;
    return kNotRegistered;
  }

  if (nullsStaged_) {
    splitStaging.addDependency(nullsStagingId_);
    return kNotRegistered;
  }
  nullsStaged_ = true;
  auto* nulls = column_->nulls.get();
  Staging staging(
      nulls->values->as<char>(),
      bits::nwords(column_->numValues) * sizeof(uint64_t),
      nulls->region);
  nullsBufferId_ = splitStaging.add(staging);
  nullsStagingId_ = splitStaging.id();
  splitStaging.registerPointer(nullsBufferId_, &grid_.nulls, true);
  return nullsBufferId_;
}

void TestFormatData::decodeAlphabet(
    ColumnOp& op,
    Column* alphabet,
    ResultStaging& deviceStaging,
    ResultStaging& resultStaging,
    SplitStaging& splitStaging,
    DecodePrograms& program,
    ReadStream& stream) {
  int32_t numRows = alphabet->numValues;
  auto rowsPerBlock = FLAGS_wave_reader_rows_per_tb;
  int32_t numBlocks = bits::roundUp(numRows, rowsPerBlock) / rowsPerBlock;
  VELOX_CHECK_LT(numBlocks, 256 * 256, "Overflow 16 bit block number");
  auto filter = op.reader->scanSpec().filter();
  BufferId filterId = kNoBufferId;
  if (filter) {
    // bitmap made of uint32_t, one bit per dictionary entry.
    filterId =
        deviceStaging.reserve(bits::roundUp(alphabet->numValues, 32) / 8);
    deviceStaging.registerPointer(filterId, &filterBitmap_, true);
  }
  BufferId decodedId = kNoBufferId;
  Staging staging(
      alphabet->values->as<char>(), alphabet->values->size(), alphabet->region);
  BufferId rawId = splitStaging.add(staging);

  for (auto blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    auto columnKind = static_cast<WaveTypeKind>(column_->kind);
    auto valueSize = waveTypeKindSize(columnKind);
    auto step = makeAlphabetStep(
        op,
        deviceStaging,
        splitStaging,
        stream,
        static_cast<WaveTypeKind>(alphabet->kind),
        blockIdx,
        numRows);
    step->resultNulls = nullptr;
    step->result = reinterpret_cast<void*>(valueSize * blockIdx * rowsPerBlock);
    if (blockIdx == 0) {
      decodedId = deviceStaging.reserve(valueSize * numRows);
      deviceStaging.registerPointer(decodedId, &decodedAlphabet_, true);
    }
    deviceStaging.registerPointer(decodedId, &step->result, false);
    step->dictMode = filter ? DictMode::kRecordFilter : DictMode::kNone;
    if (filter) {
      // Init to byte where the bitmap for this block begins.
      step->filterBitmap =
          reinterpret_cast<uint32_t*>(blockIdx * rowsPerBlock / 8);
      deviceStaging.registerPointer(filterId, &step->filterBitmap, false);
    }
    if (alphabet->encoding == Encoding::kFlat) {
      step->encoding = DecodeStep::kDictionaryOnBitpack;
      // Just bit pack, no dictionary.
      step->data.dictionaryOnBitpack.alphabet = nullptr;
      step->data.dictionaryOnBitpack.baseline = alphabet->baseline;
      step->data.dictionaryOnBitpack.bitWidth = alphabet->bitWidth;
      step->data.dictionaryOnBitpack.indices = nullptr;
      step->data.dictionaryOnBitpack.begin = 0;
      splitStaging.registerPointer(
          rawId, &step->data.dictionaryOnBitpack.indices, true);
      if (blockIdx == 0) {
        splitStaging.registerPointer(rawId, &rawAlphabet_, true);
      }
    } else {
      VELOX_NYI("Non flat alphabet encoding");
    }

    program.programs.emplace_back();
    program.programs.back().push_back(std::move(step));
  }
}

void TestFormatData::griddize(
    ColumnOp& op,
    int32_t blockSize,
    int32_t numBlocks,
    ResultStaging& deviceStaging,
    ResultStaging& resultStaging,
    SplitStaging& staging,
    DecodePrograms& programs,
    ReadStream& stream) {
  constexpr int32_t kCountStride = 1024;
  if (griddized_) {
    return;
  }
  griddized_ = true;
  if (column_->encoding == kDict) {
    auto alphabet = column_->alphabet.get();
    VELOX_CHECK_NOT_NULL(alphabet);
    VELOX_CHECK_NULL(alphabet->alphabet);
    VELOX_CHECK_NULL(alphabet->nulls);
    decodeAlphabet(
        op, alphabet, deviceStaging, resultStaging, staging, programs, stream);
  }
  if (!column_->nulls) {
    return;
  }
  // If the whole stripe is covered by a single TB, there is no need for a
  // separate griddize kernel.
  if (blockSize >= column_->numValues) {
    return;
  }
  auto id = stageNulls(deviceStaging, staging);

  auto count = std::make_unique<GpuDecode>();
  staging.registerPointer(id, &count->data.countBits.bits, true);
  auto numStrides =
      bits::roundUp(column_->numValues, kCountStride) / kCountStride;
  auto resultId = deviceStaging.reserve(sizeof(int32_t) * numStrides);
  deviceStaging.registerPointer(resultId, &count->result, true);
  deviceStaging.registerPointer(resultId, &grid_.numNonNull, true);
  count->step = DecodeStep::kCountBits;
  count->data.countBits.numBits = column_->numValues;
  count->data.countBits.resultStride = kCountStride;
  programs.programs.emplace_back();
  programs.programs.back().push_back(std::move(count));
}

void TestFormatData::stage(
    ResultStaging& deviceStaging,
    SplitStaging& splitStaging) {
  VELOX_CHECK_NOT_NULL(column_);
  // If nulls were not staged on device in griddize() they will be moved now for
  // the single TB.
  stageNulls(deviceStaging, splitStaging);
  if (values_.empty()) {
    values_.add(Staging(
        column_->values->as<char>(),
        static_cast<int32_t>(column_->values->size()),
        column_->region));
  }
  values_.stage(splitStaging);
}

void TestFormatData::setDecode(
    GpuDecode& step,
    ColumnOp& op,
    int32_t /*blockIdx*/,
    SplitStaging& splitStaging) {
  if (column_->encoding != Encoding::kFlat && column_->encoding != kDict) {
    VELOX_NYI("Unsupported test encoding");
  }
  step.encoding = DecodeStep::kDictionaryOnBitpack;
  // alphabet is set if there is a dict.
  step.dictMode =
      column_->encoding == kFlat ? DictMode::kNone : DictMode::kDict;
  auto& data = step.data.dictionaryOnBitpack;
  data.alphabet = decodedAlphabet_;
  data.baseline = column_->baseline;
  data.bitWidth = column_->bitWidth;
  data.begin = currentRow_;
  values_.setPointer(0, data.indices, 0, splitStaging);
  if (column_->encoding == kDict && op.reader->scanSpec().filter()) {
    step.dictMode = DictMode::kDictFilter;
    step.filterBitmap = filterBitmap_;
  }
}

class TestStructColumnReader : public StructColumnReader {
 public:
  TestStructColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      TestFormatParams& params,
      common::ScanSpec& scanSpec,
      Path& path,
      const DefinesMap& defines,
      bool isRoot)
      : StructColumnReader(
            requestedType,
            fileType,
            pathToOperand(defines, path),
            params,
            scanSpec,
            isRoot),
        params_(params) {
    addChildren(scanSpec, path, defines);
  }

 protected:
  std::unique_ptr<ColumnReader> makeChild(
      common::ScanSpec& childSpec,
      int32_t fileChildIdx,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      Path& path,
      const DefinesMap& defines) override {
    if (fileChildIdx < 0) {
      VELOX_NYI("Columns that are not in the file are not supported");
    }
    return TestFormatReader::build(
        requestedType, fileType, params_, childSpec, path, defines);
  }

 private:
  TestFormatParams& params_;
};

std::unique_ptr<ColumnReader> buildIntegerReader(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    TestFormatParams& params,
    common::ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines) {
  return std::make_unique<ColumnReader>(
      requestedType, fileType, pathToOperand(defines, path), params, scanSpec);
}

// static
std::unique_ptr<ColumnReader> TestFormatReader::build(
    const TypePtr& requestedType,
    const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
    TestFormatParams& params,
    common::ScanSpec& scanSpec,
    std::vector<std::unique_ptr<Subfield::PathElement>>& path,
    const DefinesMap& defines,
    bool isRoot) {
  switch (fileType->type()->kind()) {
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return buildIntegerReader(
          requestedType, fileType, params, scanSpec, path, defines);

    case TypeKind::ROW:
      return std::make_unique<TestStructColumnReader>(
          requestedType, fileType, params, scanSpec, path, defines, isRoot);
    default:
      VELOX_UNREACHABLE();
  }
}

} // namespace facebook::velox::wave::test
