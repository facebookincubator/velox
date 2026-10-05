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
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

namespace facebook::nimble {

ALPRDEncodingBase::Metadata ALPRDEncodingBase::readMetadata(
    std::string_view data,
    const Encoding::Options& options) {
  const auto prefix = EncodingPrefix::consume(data, options.useVarintRowCount);
  NIMBLE_CHECK_EQ(EncodingPrefix::encodingType(prefix), EncodingType::ALPRD);
  Metadata metadata{};
  metadata.dataType = EncodingPrefix::dataType(prefix);
  NIMBLE_CHECK(
      metadata.dataType == DataType::Float ||
          metadata.dataType == DataType::Double,
      "ALPRD requires a floating-point type.");
  metadata.rowCount =
      EncodingPrefix::readRowCount(prefix, options.useVarintRowCount);
  NIMBLE_CHECK_GT(metadata.rowCount, 0, "Empty ALPRD encoding.");
  auto& parameters = metadata.parameters;
  parameters.rightBitWidth = encoding::readByte(data);
  parameters.dictionarySize = encoding::readByte(data);
  const auto valueBitWidth = metadata.valueBitWidth();
  NIMBLE_CHECK_GE(parameters.rightBitWidth, valueBitWidth - kMaxHighBitWidth);
  NIMBLE_CHECK_LT(parameters.rightBitWidth, valueBitWidth);
  NIMBLE_CHECK_GT(parameters.dictionarySize, 0);
  NIMBLE_CHECK_LE(parameters.dictionarySize, kMaxDictionarySize);
  metadata.exceptionCount = encoding::readVarint32(data);
  NIMBLE_CHECK_LE(metadata.exceptionCount, metadata.rowCount);

  const auto highLimit = metadata.highLimit();
  for (uint8_t i = 0; i < parameters.dictionarySize; ++i) {
    const auto high = encoding::readUint16(data);
    NIMBLE_CHECK_LT(high, highLimit, "Invalid ALPRD dictionary value.");
    for (uint8_t j = 0; j < i; ++j) {
      NIMBLE_CHECK_NE(high, parameters.dictionary[j], "Duplicate ALPRD key.");
    }
    parameters.dictionary[i] = high;
  }

  const std::array<DataType, 4> childTypes{
      DataType::Uint16,
      metadata.dataType == DataType::Float ? DataType::Uint32
                                           : DataType::Uint64,
      DataType::Uint32,
      DataType::Uint16,
  };
  for (uint8_t i = 0; i < metadata.childrenCount(); ++i) {
    const auto child = encoding::readLengthPrefixedBytes(data);
    auto cursor = child;
    const auto childPrefix =
        EncodingPrefix::consume(cursor, options.useVarintRowCount);
    NIMBLE_CHECK_EQ(
        EncodingPrefix::dataType(childPrefix),
        childTypes[i],
        "Invalid ALPRD child type.");
    NIMBLE_CHECK_EQ(
        EncodingPrefix::readRowCount(childPrefix, options.useVarintRowCount),
        i < 2 ? metadata.rowCount : metadata.exceptionCount,
        "Invalid ALPRD child row count.");
    metadata.children[i] = child;
  }
  NIMBLE_CHECK(data.empty(), "Unexpected bytes after ALPRD children.");
  return metadata;
}

std::unique_ptr<Encoding> ALPRDEncodingBase::createChild(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const Encoding::Options& options) {
  auto child = EncodingFactory(options).create(
      pool, data, [](uint32_t) -> void* { return nullptr; });
  NIMBLE_CHECK(!child->isNullable(), "ALPRD children must be non-nullable.");
  return child;
}

void ALPRDEncodingBase::loadExceptions(
    velox::memory::MemoryPool& pool,
    const Metadata& metadata,
    const Encoding::Options& options,
    Vector<uint32_t>& positions,
    Vector<uint16_t>& highParts) {
  NIMBLE_DCHECK(positions.empty());
  NIMBLE_DCHECK(highParts.empty());
  if (metadata.exceptionCount == 0) {
    return;
  }
  auto positionDecoder = createChild(pool, metadata.children[2], options);
  auto highPartDecoder = createChild(pool, metadata.children[3], options);
  positions.resize(metadata.exceptionCount);
  highParts.resize(metadata.exceptionCount);
  positionDecoder->materialize(metadata.exceptionCount, positions.data());
  highPartDecoder->materialize(metadata.exceptionCount, highParts.data());
  const auto highLimit = metadata.highLimit();
  for (uint32_t i = 0; i < metadata.exceptionCount; ++i) {
    NIMBLE_CHECK_LT(
        positions[i], metadata.rowCount, "Invalid ALPRD exception position.");
    if (i != 0) {
      NIMBLE_CHECK_GT(
          positions[i],
          positions[i - 1],
          "ALPRD exception positions must increase.");
    }
    NIMBLE_CHECK_LT(
        highParts[i], highLimit, "Invalid ALPRD exception high part.");
  }
}

namespace {

// Keeps parameter training and automatic selection on the same sampled model.
struct TrainedSplit {
  ALPRDEncodingBase::Parameters parameters;
  uint64_t size{std::numeric_limits<uint64_t>::max()};
};

// Uses cheap scalar costs to bound how many times training invokes potentially
// expensive nested policies. The final score always uses the supplied policy.
struct SplitCandidate {
  ALPRDEncodingBase::Parameters parameters;
  std::array<uint64_t, 4> childSizes;
  uint32_t numExceptions;
  uint64_t size;
};

template <typename T>
uint64_t scalarChildSize(
    std::span<const T> sample,
    uint32_t numRows,
    const Encoding::Options& options) {
  const auto [min, max] = std::minmax_element(sample.begin(), sample.end());
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  if (*min == *max) {
    return prefixSize + sizeof(T);
  }
  return std::min(
      prefixSize + 1 + uint64_t{numRows} * sizeof(T),
      FixedBitWidthEncoding<T>::estimateSize(numRows, *min, *max, options) -
          EncodingPrefix::kFixedPrefixSize + prefixSize +
          FixedBitArray::bufferSize(0, 0));
}

uint64_t splitSize(
    uint8_t dictionarySize,
    uint32_t numRows,
    uint32_t numExceptions,
    const std::array<uint64_t, 4>& childSizes,
    const Encoding::Options& options) {
  uint64_t size =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount) + 2 +
      2 * dictionarySize + varint::varintSize(numExceptions);
  for (auto childSize : childSizes) {
    if (childSize != 0) {
      size += varint::varintSize(childSize) + childSize;
    }
  }
  return size;
}

// Owns the bounded sample and scratch storage for one ALPRD training run.
template <typename PhysicalType>
class SplitTraining {
 public:
  // Samples the input while retaining positions in the target full stream.
  SplitTraining(
      std::span<const PhysicalType> values,
      uint32_t numRows,
      const Encoding::Options& options,
      const ALPRDEncodingBase::ChildPolicies& childPolicies);

  // Shortlists splits with scalar costs, then scores their selected children.
  TrainedSplit train();

 private:
  using Base = ALPRDEncodingBase;

  // Number of values represented by the sample in the target payload.
  const uint32_t numRows_;
  // Samples remain bounded even when the input span contains all values.
  const uint32_t numSamples_;
  // Borrows the caller's options and child policies for this training run.
  const Encoding::Options& options_;
  const Base::ChildPolicies& childPolicies_;
  // Original sampled bits and their mapped positions in the target payload.
  std::array<PhysicalType, Base::kSampleSize> sample_{};
  std::array<uint32_t, Base::kSampleSize> samplePositions_{};
  // Reuses split components and frequency storage across candidate widths.
  std::array<PhysicalType, Base::kSampleSize> rightParts_{};
  std::array<uint16_t, Base::kSampleSize> highParts_{};
  std::array<uint16_t, Base::kSampleSize> sortedHighParts_{};
  std::array<std::pair<uint16_t, uint32_t>, Base::kSampleSize> frequencies_{};
  // Holds the sampled codes and exceptions for the candidate being scored.
  std::array<uint16_t, Base::kSampleSize> codes_{};
  std::array<uint32_t, Base::kSampleSize> exceptionPositions_{};
  std::array<uint16_t, Base::kSampleSize> exceptionHighParts_{};
};

template <typename PhysicalType>
SplitTraining<PhysicalType>::SplitTraining(
    std::span<const PhysicalType> values,
    uint32_t numRows,
    const Encoding::Options& options,
    const ALPRDEncodingBase::ChildPolicies& childPolicies)
    : numRows_{numRows},
      numSamples_{std::min<uint32_t>(values.size(), Base::kSampleSize)},
      options_{options},
      childPolicies_{childPolicies} {
  NIMBLE_CHECK(!values.empty(), "ALPRD training requires non-empty input.");
  NIMBLE_CHECK_LE(values.size(), numRows_);
  for (uint32_t i = 0; i < numSamples_; ++i) {
    const auto index = detail::NestedAlpSizeEstimation::sampledRowIndex(
        i, numSamples_, values.size());
    sample_[i] = values[index];
    samplePositions_[i] = uint64_t{index} * numRows_ / values.size();
  }
}

template <typename PhysicalType>
TrainedSplit SplitTraining<PhysicalType>::train() {
  std::vector<SplitCandidate> candidates;
  candidates.reserve(Base::kMaxHighBitWidth * Base::kMaxDictionarySize);
  for (uint8_t highBitWidth = 1; highBitWidth <= Base::kMaxHighBitWidth;
       ++highBitWidth) {
    const uint8_t rightBitWidth = sizeof(PhysicalType) * 8 - highBitWidth;
    const auto mask = (PhysicalType{1} << rightBitWidth) - 1;
    for (uint32_t i = 0; i < numSamples_; ++i) {
      rightParts_[i] = sample_[i] & mask;
      highParts_[i] = sample_[i] >> rightBitWidth;
      sortedHighParts_[i] = highParts_[i];
    }
    std::sort(sortedHighParts_.begin(), sortedHighParts_.begin() + numSamples_);
    uint32_t numPrefixes{0};
    for (uint32_t i = 0; i < numSamples_; ++i) {
      if (i == 0 || sortedHighParts_[i] != sortedHighParts_[i - 1]) {
        frequencies_[numPrefixes++] = {sortedHighParts_[i], 1};
      } else {
        ++frequencies_[numPrefixes - 1].second;
      }
    }
    const auto maxDictionarySize =
        std::min<uint32_t>(numPrefixes, Base::kMaxDictionarySize);
    std::partial_sort(
        frequencies_.begin(),
        frequencies_.begin() + maxDictionarySize,
        frequencies_.begin() + numPrefixes,
        [](const auto& lhs, const auto& rhs) {
          return lhs.second != rhs.second ? lhs.second > rhs.second
                                          : lhs.first < rhs.first;
        });
    const auto rightSize = scalarChildSize<PhysicalType>(
        {rightParts_.data(), numSamples_}, numRows_, options_);
    for (uint8_t dictionarySize = 1; dictionarySize <= maxDictionarySize;
         ++dictionarySize) {
      uint32_t sampleExceptions{0};
      for (uint32_t i = 0; i < numSamples_; ++i) {
        uint16_t code{0};
        while (code < dictionarySize &&
               frequencies_[code].first != highParts_[i]) {
          ++code;
        }
        if (code == dictionarySize) {
          exceptionPositions_[sampleExceptions] = samplePositions_[i];
          exceptionHighParts_[sampleExceptions++] = highParts_[i];
          code = 0;
        }
        codes_[i] = code;
      }
      const uint32_t numExceptions =
          (uint64_t{sampleExceptions} * numRows_ + numSamples_ - 1) /
          numSamples_;
      const std::array<uint64_t, 4> childSizes{
          scalarChildSize<uint16_t>(
              {codes_.data(), numSamples_}, numRows_, options_),
          rightSize,
          numExceptions == 0
              ? 0
              : scalarChildSize<uint32_t>(
                    {exceptionPositions_.data(), sampleExceptions},
                    numExceptions,
                    options_),
          numExceptions == 0
              ? 0
              : scalarChildSize<uint16_t>(
                    {exceptionHighParts_.data(), sampleExceptions},
                    numExceptions,
                    options_),
      };
      Base::Parameters parameters{
          .rightBitWidth = rightBitWidth, .dictionarySize = dictionarySize};
      for (uint8_t i = 0; i < dictionarySize; ++i) {
        parameters.dictionary[i] = frequencies_[i].first;
      }
      // Equivalent scalar layouts must not crowd out other split shapes. On
      // ties keep the narrower right part, as in the final scoring below.
      const auto equivalent = std::find_if(
          candidates.begin(), candidates.end(), [&](const auto& candidate) {
            return candidate.parameters.dictionarySize == dictionarySize &&
                candidate.numExceptions == numExceptions &&
                candidate.childSizes == childSizes;
          });
      if (equivalent != candidates.end()) {
        equivalent->parameters = parameters;
      } else {
        candidates.push_back(
            {parameters,
             childSizes,
             numExceptions,
             splitSize(
                 dictionarySize,
                 numRows_,
                 numExceptions,
                 childSizes,
                 options_)});
      }
    }
  }
  constexpr uint32_t kMaxCandidates = 4;
  const auto numCandidates =
      std::min<uint32_t>(candidates.size(), kMaxCandidates);
  std::partial_sort(
      candidates.begin(),
      candidates.begin() + numCandidates,
      candidates.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.size != rhs.size ? lhs.size < rhs.size
            : lhs.parameters.rightBitWidth != rhs.parameters.rightBitWidth
            ? lhs.parameters.rightBitWidth < rhs.parameters.rightBitWidth
            : lhs.parameters.dictionarySize < rhs.parameters.dictionarySize;
      });
  TrainedSplit best;
  for (uint32_t candidateIndex = 0; candidateIndex < numCandidates;
       ++candidateIndex) {
    const auto& candidate = candidates[candidateIndex];
    const auto& parameters = candidate.parameters;
    const auto mask = (PhysicalType{1} << parameters.rightBitWidth) - 1;
    uint32_t sampleExceptions{0};
    for (uint32_t i = 0; i < numSamples_; ++i) {
      rightParts_[i] = sample_[i] & mask;
      const uint16_t high = sample_[i] >> parameters.rightBitWidth;
      uint16_t code{0};
      while (code < parameters.dictionarySize &&
             parameters.dictionary[code] != high) {
        ++code;
      }
      if (code == parameters.dictionarySize) {
        exceptionPositions_[sampleExceptions] = samplePositions_[i];
        exceptionHighParts_[sampleExceptions++] = high;
        code = 0;
      }
      codes_[i] = code;
    }
    const auto numExceptions = candidate.numExceptions;
    const std::array<uint64_t, 4> childSizes{
        detail::NestedAlpSizeEstimation::estimateChildSize<uint16_t>(
            {codes_.data(), numSamples_},
            numRows_,
            options_,
            *childPolicies_.codes),
        detail::NestedAlpSizeEstimation::estimateChildSize<PhysicalType>(
            {rightParts_.data(), numSamples_},
            numRows_,
            options_,
            *childPolicies_.rightParts),
        numExceptions == 0
            ? 0
            : detail::NestedAlpSizeEstimation::estimateChildSize<uint32_t>(
                  {exceptionPositions_.data(), sampleExceptions},
                  numExceptions,
                  options_,
                  *childPolicies_.exceptionPositions),
        numExceptions == 0
            ? 0
            : detail::NestedAlpSizeEstimation::estimateChildSize<uint16_t>(
                  {exceptionHighParts_.data(), sampleExceptions},
                  numExceptions,
                  options_,
                  *childPolicies_.exceptionHighParts),
    };
    const auto size = splitSize(
        parameters.dictionarySize,
        numRows_,
        numExceptions,
        childSizes,
        options_);
    if (size < best.size ||
        (size == best.size &&
         parameters.rightBitWidth < best.parameters.rightBitWidth)) {
      best = {parameters, size};
    }
  }
  return best;
}

template <typename PhysicalType>
TrainedSplit trainSplit(
    std::span<const PhysicalType> values,
    uint32_t numRows,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  NIMBLE_CHECK(!values.empty(), "ALPRD training requires non-empty input.");
  NIMBLE_CHECK_LE(values.size(), numRows);
  std::unique_ptr<EncodingSelectionPolicyBase> defaultPolicy;
  if (policy == nullptr) {
    defaultPolicy =
        ManualEncodingSelectionPolicyFactory{
            ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
            std::nullopt}
            .createPolicy(TypeTraits<PhysicalType>::dataType);
    policy = defaultPolicy.get();
  }
  return SplitTraining<PhysicalType>{
      values,
      numRows,
      options,
      ALPRDEncodingBase::ChildPolicies{
          .codes = policy->create<uint16_t>(
              EncodingType::ALPRD, EncodingIdentifiers::ALPRD::Codes),
          .rightParts = policy->create<PhysicalType>(
              EncodingType::ALPRD, EncodingIdentifiers::ALPRD::RightParts),
          .exceptionPositions = policy->create<uint32_t>(
              EncodingType::ALPRD,
              EncodingIdentifiers::ALPRD::ExceptionPositions),
          .exceptionHighParts = policy->create<uint16_t>(
              EncodingType::ALPRD,
              EncodingIdentifiers::ALPRD::ExceptionHighParts),
      }}
      .train();
}

} // namespace

template <typename PhysicalType>
ALPRDEncodingBase::Parameters ALPRDEncodingBase::selectParameters(
    std::span<const PhysicalType> values,
    const Encoding::Options& options,
    const ChildPolicies& childPolicies) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return SplitTraining<PhysicalType>{
      values, static_cast<uint32_t>(values.size()), options, childPolicies}
      .train()
      .parameters;
}

template <typename PhysicalType>
ALPRDEncodingBase::Parameters ALPRDEncodingBase::selectParameters(
    std::span<const PhysicalType> values,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return trainSplit(values, values.size(), options, policy).parameters;
}

template <typename PhysicalType>
std::optional<uint64_t> ALPRDEncodingBase::estimateSize(
    std::span<const PhysicalType> sampleValues,
    uint32_t numRows,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  if (sampleValues.empty()) {
    return std::nullopt;
  }
  return trainSplit(sampleValues, numRows, options, policy).size;
}

template ALPRDEncodingBase::Parameters
ALPRDEncodingBase::selectParameters<uint32_t>(
    std::span<const uint32_t>,
    const Encoding::Options&,
    const ALPRDEncodingBase::ChildPolicies&);
template ALPRDEncodingBase::Parameters
ALPRDEncodingBase::selectParameters<uint64_t>(
    std::span<const uint64_t>,
    const Encoding::Options&,
    const ALPRDEncodingBase::ChildPolicies&);
template ALPRDEncodingBase::Parameters
ALPRDEncodingBase::selectParameters<uint32_t>(
    std::span<const uint32_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template ALPRDEncodingBase::Parameters
ALPRDEncodingBase::selectParameters<uint64_t>(
    std::span<const uint64_t>,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template std::optional<uint64_t> ALPRDEncodingBase::estimateSize<uint32_t>(
    std::span<const uint32_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);
template std::optional<uint64_t> ALPRDEncodingBase::estimateSize<uint64_t>(
    std::span<const uint64_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase*);

} // namespace facebook::nimble
