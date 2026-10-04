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

#include "velox/dwio/nimble/encodings/ALPEncoding.h"

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

namespace facebook::nimble {

template <typename T>
std::optional<uint64_t> ALPEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  if (values.empty()) {
    return std::nullopt;
  }

  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  const uint64_t rowCount = values.size();
  const uint32_t sampleSize = estimateSampleSize(rowCount);

  std::vector<physicalType> sampledValues;
  sampledValues.reserve(sampleSize);
  // Vary the offset within each interval so periodic inputs retain their
  // distinct values when estimating dictionary and run-length children.
  for (uint32_t i = 0; i < sampleSize; ++i) {
    const auto inputIndex = detail::NestedAlpSizeEstimation::sampledRowIndex(
        i, sampleSize, rowCount);
    sampledValues.push_back(values[inputIndex]);
  }

  return estimateSizeFromSample(rowCount, sampledValues, options, policy);
}

template <typename T>
std::optional<uint64_t> ALPEncoding<T>::estimateSizeFromSample(
    uint64_t rowCount,
    std::span<const physicalType> sampledValues,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  NIMBLE_CHECK_GT(rowCount, 0, "ALP estimation requires non-empty input.");
  NIMBLE_CHECK(
      !sampledValues.empty(), "ALP estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(
      sampledValues.size(),
      rowCount,
      "ALP sample size cannot exceed the input row count.");

  NIMBLE_CHECK_LE(rowCount, std::numeric_limits<uint32_t>::max());
  const uint64_t sampleSize = sampledValues.size();

  std::vector<cppDataType> logicalValues;
  logicalValues.reserve(sampleSize);
  for (const auto value : sampledValues) {
    logicalValues.push_back(detail::alp::toLogical<cppDataType>(value));
  }

  const auto [exponent, factor] = findBestExponentFactorByCount(
      std::span<const cppDataType>{logicalValues.data(), logicalValues.size()});

  std::vector<uint64_t> encodedValues;
  encodedValues.reserve(sampleSize);
  std::vector<uint32_t> exceptionPositions;
  exceptionPositions.reserve(sampleSize);
  std::vector<physicalType> exceptionValues;
  exceptionValues.reserve(sampleSize);
  uint64_t sampleExceptionCount{0};
  for (auto i = 0; i < sampleSize; ++i) {
    if (!canRepresentExactly(
            logicalValues[i],
            sampledValues[static_cast<size_t>(i)],
            exponent,
            factor)) {
      encodedValues.push_back(0);
      exceptionPositions.push_back(sampledValueIndex(i, rowCount, sampleSize));
      exceptionValues.push_back(sampledValues[static_cast<size_t>(i)]);
      ++sampleExceptionCount;
      continue;
    }

    const auto encoded =
        encodeValue(static_cast<double>(logicalValues[i]), exponent, factor);
    encodedValues.push_back(velox::ZigZag::encode(encoded));
  }

  std::unique_ptr<EncodingSelectionPolicyBase> defaultPolicy;
  if (policy == nullptr) {
    defaultPolicy =
        ManualEncodingSelectionPolicyFactory{
            ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
            std::nullopt}
            .createPolicy(TypeTraits<T>::dataType);
    policy = defaultPolicy.get();
  }
  auto encodedPolicy = policy->create<uint64_t>(
      EncodingType::ALP, EncodingIdentifiers::ALP::EncodedValues);
  const uint64_t nestedEncodedValuesSize =
      detail::NestedAlpSizeEstimation::estimateChildSize<uint64_t>(
          encodedValues, rowCount, options, *encodedPolicy);
  const uint64_t exceptionCount =
      (sampleExceptionCount * rowCount + sampleSize - 1) / sampleSize;
  uint64_t exceptionPositionsSize{0};
  uint64_t exceptionValuesSize{0};
  if (exceptionCount > 0) {
    auto positionsPolicy = policy->create<uint32_t>(
        EncodingType::ALP, EncodingIdentifiers::ALP::ExceptionPositions);
    exceptionPositionsSize =
        detail::NestedAlpSizeEstimation::estimateChildSize<uint32_t>(
            exceptionPositions, exceptionCount, options, *positionsPolicy);
    auto valuesPolicy = policy->create<physicalType>(
        EncodingType::ALP, EncodingIdentifiers::ALP::ExceptionValues);
    exceptionValuesSize =
        detail::NestedAlpSizeEstimation::estimateChildSize<physicalType>(
            exceptionValues, exceptionCount, options, *valuesPolicy);
  }
  const uint64_t metadataSize = kHeaderSize +
      (exceptionCount > 0 ? varint::varintSize(exceptionCount) : 0) +
      varint::varintSize(nestedEncodedValuesSize) +
      (exceptionCount > 0 ? varint::varintSize(exceptionPositionsSize) +
               varint::varintSize(exceptionValuesSize)
                          : 0);
  return Encoding::serializePrefixSize(
             static_cast<uint32_t>(rowCount), options.useVarintRowCount) +
      metadataSize + nestedEncodedValuesSize + exceptionPositionsSize +
      exceptionValuesSize;
}

template <typename T>
void ALPEncoding<T>::decodeBulkValues(
    const uint64_t* encodedValues,
    vector_size_t numValues,
    int exponent,
    int factor,
    cppDataType* output) {
  using UnsignedBatch = xsimd::batch<uint64_t>;
  using SignedBatch = xsimd::batch<int64_t>;
  using DoubleBatch = xsimd::batch<double>;
  constexpr auto kDecodeBatchSize =
      static_cast<vector_size_t>(DoubleBatch::size);
  const DoubleBatch exponentMultiplier(kPow10Double[exponent]);
  const DoubleBatch factorMultiplier(kPow10Double[factor]);
  vector_size_t row = 0;
  for (; row <= numValues - kDecodeBatchSize; row += kDecodeBatchSize) {
    // ALP stores transformed values as ZigZag-encoded uint64_t lanes for both
    // float and double inputs. Shifting removes the sign bit; XOR with the
    // zero or all-ones sign mask restores each signed integer.
    const auto zigZag = UnsignedBatch::load_unaligned(encodedValues + row);
    const auto signedBits =
        (zigZag >> 1) ^ (UnsignedBatch(0) - (zigZag & UnsignedBatch(1)));
    const auto integers = xsimd::bitwise_cast<SignedBatch>(signedBits);
    const auto restored = xsimd::batch_cast<double>(integers) *
        factorMultiplier / exponentMultiplier;
    restored.store_unaligned(output + row);
  }
  for (; row < numValues; ++row) {
    output[row] = decodeValue(
        velox::ZigZag::decode(encodedValues[row]), exponent, factor);
  }
}

template void ALPEncoding<float>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    float*);
template void ALPEncoding<double>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    double*);

template std::optional<uint64_t> ALPEncoding<float>::estimateSize(
    std::span<const uint32_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template std::optional<uint64_t> ALPEncoding<float>::estimateSizeFromSample(
    uint64_t,
    std::span<const uint32_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template std::optional<uint64_t> ALPEncoding<double>::estimateSize(
    std::span<const uint64_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template std::optional<uint64_t> ALPEncoding<double>::estimateSizeFromSample(
    uint64_t,
    std::span<const uint64_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);

} // namespace facebook::nimble
