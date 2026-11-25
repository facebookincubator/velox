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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::wave {
namespace {

GpuDecode selective(
    DecodeStep step,
    WaveFilterKind filter,
    NullMode nullMode,
    DictMode dictMode = DictMode::kNone) {
  GpuDecode op;
  op.step = step;
  op.encoding = DecodeStep::kDictionaryOnBitpack;
  op.filterKind = filter;
  op.nullMode = nullMode;
  op.dictMode = dictMode;
  op.result = &op;
  return op;
}

GpuDecode trivial(WaveTypeKind type) {
  GpuDecode op;
  op.step = DecodeStep::kTrivial;
  op.data.trivial.dataType = type;
  return op;
}

std::vector<GpuDecode*> pointers(std::vector<GpuDecode>& ops) {
  std::vector<GpuDecode*> result;
  result.reserve(ops.size());
  for (auto& op : ops) {
    result.push_back(&op);
  }
  return result;
}

TEST(ReaderKernelTest, keyIndependentOfOpOrder) {
  std::vector<GpuDecode> ops = {
      selective(
          DecodeStep::kSelective64,
          WaveFilterKind::kBigintRange,
          NullMode::kDenseNonNull),
      selective(
          DecodeStep::kSelective32,
          WaveFilterKind::kAlwaysTrue,
          NullMode::kSparseNullable),
      trivial(WaveTypeKind::BIGINT),
      selective(
          DecodeStep::kSelective64,
          WaveFilterKind::kBigintRange,
          NullMode::kDenseNonNull),
  };
  std::vector<GpuDecode> reversed(ops.rbegin(), ops.rend());

  auto key = makeReaderKernelKey(pointers(ops), 256);
  auto reversedKey = makeReaderKernelKey(pointers(reversed), 256);

  EXPECT_EQ(key.toString(), reversedKey.toString());
  EXPECT_EQ(3, key.signatures.size());
  EXPECT_EQ(ops[0].opcode, ops[3].opcode);
  EXPECT_EQ(ops[0].opcode, reversed[0].opcode);
  EXPECT_EQ(ops[1].opcode, reversed[2].opcode);
  EXPECT_EQ(ops[2].opcode, reversed[1].opcode);
  EXPECT_NE(ops[0].opcode, ops[1].opcode);
}

TEST(ReaderKernelTest, keyDistinguishesInstantiations) {
  auto keyOf = [](GpuDecode op, int32_t blockSize = 256) {
    std::vector<GpuDecode> ops = {op};
    return makeReaderKernelKey(pointers(ops), blockSize).toString();
  };
  auto base = selective(
      DecodeStep::kSelective64,
      WaveFilterKind::kBigintRange,
      NullMode::kDenseNonNull);
  auto noResult = base;
  noResult.result = nullptr;
  auto nullable = base;
  nullable.nullMode = NullMode::kDenseNullable;
  auto recordFilter = base;
  recordFilter.dictMode = DictMode::kRecordFilter;
  auto narrow = base;
  narrow.step = DecodeStep::kSelective32;

  std::vector<std::string> keys = {
      keyOf(base),
      keyOf(noResult),
      keyOf(nullable),
      keyOf(recordFilter),
      keyOf(narrow),
      keyOf(base, 128),
  };
  std::sort(keys.begin(), keys.end());
  EXPECT_EQ(keys.end(), std::unique(keys.begin(), keys.end()));
}

TEST(ReaderKernelTest, signatureNormalization) {
  // REAL and INTEGER decode with the same instantiation.
  EXPECT_EQ(
      decodeSignature(trivial(WaveTypeKind::REAL)),
      decodeSignature(trivial(WaveTypeKind::INTEGER)));

  // A result pointer does not matter without a filter.
  auto withResult = selective(
      DecodeStep::kSelective64,
      WaveFilterKind::kAlwaysTrue,
      NullMode::kDenseNonNull);
  auto withoutResult = withResult;
  withoutResult.result = nullptr;
  EXPECT_EQ(decodeSignature(withResult), decodeSignature(withoutResult));

  // A dictionary filter replaces the filter kind and decodes 32 bit indices.
  auto dictFilter = selective(
      DecodeStep::kSelective64,
      WaveFilterKind::kBigintRange,
      NullMode::kDenseNonNull,
      DictMode::kDictFilter);
  auto signature = decodeSignature(dictFilter);
  EXPECT_EQ(WaveFilterKind::kDictFilter, signature.filterKind);
  EXPECT_EQ(WaveTypeKind::INTEGER, signature.indexType);
  EXPECT_EQ(WaveTypeKind::BIGINT, signature.dataType);
}

TEST(ReaderKernelTest, unsupportedOps) {
  auto doubleRange = selective(
      DecodeStep::kSelective64,
      WaveFilterKind::kDoubleRange,
      NullMode::kDenseNonNull);
  VELOX_ASSERT_THROW(
      decodeSignature(doubleRange),
      "Filter kind has no compiled reader implementation");

  GpuDecode varint;
  varint.step = DecodeStep::kVarint;
  varint.data.varint.resultType = WaveTypeKind::SMALLINT;
  VELOX_ASSERT_THROW(
      decodeSignature(varint), "kVarint only supports INTEGER and BIGINT");
}

TEST(ReaderKernelTest, generatedSource) {
  std::vector<GpuDecode> ops = {
      selective(
          DecodeStep::kSelective64,
          WaveFilterKind::kBigintRange,
          NullMode::kSparseNullable,
          DictMode::kRecordFilter),
      trivial(WaveTypeKind::DOUBLE),
  };
  auto spec = readerKernelSpec(makeReaderKernelKey(pointers(ops), 256));

  EXPECT_THAT(
      spec.code,
      testing::HasSubstr(
          "detail::decodeSelective<int64_t, 256, DecodeStep::kDictionaryOnBitpack, "
          "WaveFilterKind::kBigintRange, true, int64_t, NullMode::kSparseNullable, "
          "DictMode::kRecordFilter>(&op);"));
  EXPECT_THAT(
      spec.code,
      testing::HasSubstr("detail::decodeTrivial<uint64_t>(op.data.trivial);"));
  EXPECT_THAT(spec.code, testing::HasSubstr("switch (op.opcode)"));
  EXPECT_EQ(2, spec.entryPoints.size());
}

} // namespace
} // namespace facebook::velox::wave
