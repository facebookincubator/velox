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

#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"

#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"

namespace facebook::nimble {

// ===========================================================================
// Leaf encoding conversions
// ===========================================================================

TrivialEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::Trivial, {}, CompressionType::Uncompressed);
}

StringTrivialEnc::operator EncodingLayout() const {
  constexpr auto kLengths = EncodingIdentifiers::Trivial::Lengths;
  static_assert(kLengths == 0);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(1);
  children.emplace_back(lengths);
  return EncodingLayout(
      EncodingType::Trivial,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

ConstantEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::Constant, {}, CompressionType::Uncompressed);
}

FixedBitWidthEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed);
}

VarintEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::Varint, {}, CompressionType::Uncompressed);
}

SparseBoolEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::SparseBool, {}, CompressionType::Uncompressed);
}

BlockBitPackingEnc::operator EncodingLayout() const {
  return EncodingLayout(
      EncodingType::BlockBitPacking,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt, std::nullopt});
}

DeltaEnc::operator EncodingLayout() const {
  constexpr auto kDeltas = EncodingIdentifiers::Delta::Deltas;
  constexpr auto kRestatements = EncodingIdentifiers::Delta::Restatements;
  constexpr auto kIsRestatements = EncodingIdentifiers::Delta::IsRestatements;
  static_assert(kDeltas == 0 && kRestatements == 1 && kIsRestatements == 2);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(3);
  children.emplace_back(deltas);
  children.emplace_back(restatements);
  children.emplace_back(isRestatements);
  return EncodingLayout(
      EncodingType::Delta,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

// ===========================================================================
// Compound encoding conversions
// ===========================================================================

RLEEnc::operator EncodingLayout() const {
  constexpr auto kRunLengths = EncodingIdentifiers::RunLength::RunLengths;
  constexpr auto kRunValues = EncodingIdentifiers::RunLength::RunValues;
  static_assert(kRunLengths == 0 && kRunValues == 1);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(2);
  children.emplace_back(runLengths);
  children.emplace_back(runValues);
  return EncodingLayout(
      EncodingType::RLE,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

DictionaryEnc::operator EncodingLayout() const {
  constexpr auto kAlphabet = EncodingIdentifiers::Dictionary::Alphabet;
  constexpr auto kIndices = EncodingIdentifiers::Dictionary::Indices;
  static_assert(kAlphabet == 0 && kIndices == 1);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(2);
  children.emplace_back(alphabet);
  children.emplace_back(indices);
  return EncodingLayout(
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

MainlyConstantEnc::operator EncodingLayout() const {
  constexpr auto kIsCommon = EncodingIdentifiers::MainlyConstant::IsCommon;
  constexpr auto kOtherValues =
      EncodingIdentifiers::MainlyConstant::OtherValues;
  static_assert(kIsCommon == 0 && kOtherValues == 1);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(2);
  children.emplace_back(isCommon);
  children.emplace_back(otherValues);
  return EncodingLayout(
      EncodingType::MainlyConstant,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

NullableEnc::operator EncodingLayout() const {
  constexpr auto kData = EncodingIdentifiers::Nullable::Data;
  constexpr auto kNulls = EncodingIdentifiers::Nullable::Nulls;
  static_assert(kData == 0 && kNulls == 1);

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(2);
  children.emplace_back(data);
  children.emplace_back(nulls);
  return EncodingLayout(
      EncodingType::Nullable,
      {},
      CompressionType::Uncompressed,
      std::move(children));
}

} // namespace facebook::nimble
