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

#include <folly/hash/Hash.h>

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

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

uint32_t
sampledRowIndex(uint32_t sampleIndex, uint32_t numSamples, uint32_t numRows) {
  NIMBLE_DCHECK_LT(sampleIndex, numSamples);
  NIMBLE_DCHECK_LE(numSamples, numRows);
  const auto begin = uint64_t{sampleIndex} * numRows / numSamples;
  const auto end = uint64_t{sampleIndex + 1} * numRows / numSamples;
  return begin + folly::hash::twang_mix64(sampleIndex + 1) % (end - begin);
}

// Estimates ALPRD child streams and floating-point children that can select
// ALPRD. Sampling state and projected costs stay local to this training model.
template <typename T>
class SampledCost {
 public:
  using PhysicalType = typename TypeTraits<T>::physicalType;

  // Statistics describe the observed sample; numTotalRows is the target size.
  SampledCost(
      EncodingSelectionPolicyBase& policy,
      std::span<const PhysicalType> sampleValues,
      uint32_t numTotalRows,
      const Encoding::Options& options);

  // Scores manual candidates at full size or honors a policy-selected layout.
  uint64_t selectedSize();

 private:
  // A supplied policy also estimates explicitly bound container children.
  std::optional<uint64_t> estimateSize(
      EncodingType encodingType,
      EncodingSelectionPolicyBase* policy);

  // Borrows the child policy that carries candidate and layout restrictions.
  EncodingSelectionPolicy<T>& policy_;
  // Observed values and the full stream length they represent.
  const std::span<const PhysicalType> sampleValues_;
  const uint32_t numTotalRows_;
  // Builds statistics once for all candidate estimates.
  const Statistics<PhysicalType> statistics_;
  // Prefix and bit-packing options must match the subsequent writer.
  const Encoding::Options& options_;
};

// Samples the derived sequence without allocating a full run-value or uncommon
// value stream. The caller already knows its length from the cached statistics.
template <typename T, typename Predicate>
std::span<const T> sampleFilteredValues(
    std::span<const T> values,
    uint32_t numValues,
    Predicate include,
    std::array<T, ALPRDEncodingBase::kSampleSize>& storage) {
  const auto sampleSize = std::min<uint32_t>(numValues, storage.size());
  if (sampleSize == 0) {
    return {};
  }
  uint32_t ordinal{0};
  uint32_t sampled{0};
  auto nextIndex = sampledRowIndex(0, sampleSize, numValues);
  for (const auto value : values) {
    if (!include(value)) {
      continue;
    }
    if (ordinal++ == nextIndex) {
      storage[sampled++] = value;
      if (sampled == sampleSize) {
        break;
      }
      nextIndex = sampledRowIndex(sampled, sampleSize, numValues);
    }
  }
  return {storage.data(), sampled};
}

// Estimates a floating-point container using its actual value-child policy.
// Returns nullopt for encodings other than Dictionary, RLE and MainlyConstant.
template <typename T>
std::optional<uint64_t> estimateNestedFloatingPointSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    uint32_t numRows,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    EncodingSelectionPolicyBase& policy,
    const Encoding::Options& options) {
  static_assert(isFloatingPointType<T>());
  using PhysicalType = typename TypeTraits<T>::physicalType;
  if (encodingType != EncodingType::Dictionary &&
      encodingType != EncodingType::RLE &&
      encodingType != EncodingType::MainlyConstant) {
    return std::nullopt;
  }
  const auto scaleCount = [&](uint64_t count) -> uint32_t {
    return (count * numRows + values.size() - 1) / values.size();
  };
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  std::array<PhysicalType, ALPRDEncodingBase::kSampleSize> storage;
  std::span<const PhysicalType> sample;
  uint32_t numChildRows{0};
  NestedEncodingIdentifier identifier;
  uint64_t otherSize{0};
  switch (encodingType) {
    case EncodingType::Dictionary: {
      const auto& counts = statistics.uniqueCounts().value();
      // A dictionary alphabet is a set, so count each sampled key once rather
      // than retaining the original values' frequency weights.
      uint32_t sampled{0};
      for (const auto& [value, count] : counts) {
        storage[sampled++] = value;
        if (sampled == storage.size()) {
          break;
        }
      }
      sample = {storage.data(), sampled};
      numChildRows = scaleCount(counts.size());
      identifier = EncodingIdentifiers::Dictionary::Alphabet;
      otherSize = prefixSize + sizeof(uint32_t) +
          FixedBitWidthEncoding<uint32_t>::estimateSize(
                      numRows, 0, numChildRows - 1, options);
      break;
    }
    case EncodingType::RLE: {
      const auto numRuns = statistics.consecutiveRepeatCount();
      std::optional<PhysicalType> previous;
      sample = sampleFilteredValues(
          values,
          numRuns,
          [&](PhysicalType value) {
            if (previous == value) {
              return false;
            }
            previous = value;
            return true;
          },
          storage);
      numChildRows = scaleCount(numRuns);
      identifier = EncodingIdentifiers::RunLength::RunValues;
      otherSize = prefixSize + sizeof(uint32_t) +
          FixedBitWidthEncoding<uint32_t>::estimateSize(
                      numChildRows,
                      statistics.minRepeat(),
                      statistics.maxRepeat(),
                      options);
      break;
    }
    case EncodingType::MainlyConstant: {
      const auto common = statistics.uniqueCounts()->mostFrequent().value();
      const auto numUncommon = values.size() - common.second;
      if (numUncommon != 0) {
        sample = sampleFilteredValues(
            values,
            numUncommon,
            [&](PhysicalType value) { return value != common.first; },
            storage);
      }
      numChildRows = scaleCount(numUncommon);
      identifier = EncodingIdentifiers::MainlyConstant::OtherValues;
      otherSize = prefixSize + 2 * sizeof(uint32_t) + sizeof(PhysicalType) +
          SparseBoolEncoding::estimateSize(numRows, numChildRows, options);
      break;
    }
    default:
      NIMBLE_UNREACHABLE("Unexpected floating-point container.");
  }
  auto childPolicy = policy.create<T>(encodingType, identifier);
  return otherSize +
      SampledCost<T>{*childPolicy, sample, numChildRows, options}
          .selectedSize();
}

template <typename T>
SampledCost<T>::SampledCost(
    EncodingSelectionPolicyBase& policy,
    std::span<const PhysicalType> sampleValues,
    uint32_t numTotalRows,
    const Encoding::Options& options)
    : policy_{static_cast<EncodingSelectionPolicy<T>&>(policy)},
      sampleValues_{sampleValues},
      numTotalRows_{numTotalRows},
      statistics_{Statistics<PhysicalType>::create(sampleValues)},
      options_{options} {}

template <typename T>
uint64_t SampledCost<T>::selectedSize() {
  EncodingType selectedEncoding{EncodingType::Trivial};
  std::optional<uint64_t> size;
  if (const auto* manual =
          dynamic_cast<const ManualEncodingSelectionPolicy<T>*>(&policy_)) {
    // Compare every candidate after projecting its cost to the target stream.
    // Sampled candidates retain the existing container heuristics to bound
    // recursive training, even when the sample contains every input row.
    NIMBLE_CHECK_LE(sampleValues_.size(), numTotalRows_);
    if (!sampleValues_.empty()) {
      float minCost = std::numeric_limits<float>::max();
      for (const auto& [encodingType, readFactor] :
           manual->candidateEncodingReadFactors(options_)) {
        const auto estimatedSize = estimateSize(
            encodingType,
            encodingType == EncodingType::ALPRD ? &policy_ : nullptr);
        if (!estimatedSize.has_value()) {
          continue;
        }
        const auto cost = *estimatedSize * readFactor;
        if (cost < minCost) {
          minCost = cost;
          selectedEncoding = encodingType;
          size = estimatedSize;
        }
      }
    }
  } else {
    const auto result = policy_.select(sampleValues_, statistics_, options_);
    selectedEncoding = result.encodingType;
    // A policy's estimate describes its input, so reuse it only when the
    // sample covers the full stream. The selected layout remains binding.
    if (numTotalRows_ == sampleValues_.size()) {
      size = result.estimatedSize;
    }
  }
  if (!size) {
    size = estimateSize(selectedEncoding, &policy_);
  }
  const auto prefixSize =
      EncodingPrefix::serializedSize(numTotalRows_, options_.useVarintRowCount);
  if (!size) {
    // Custom policies can select codecs without estimators. Keep their layout
    // binding and use an uncompressed size as the training approximation.
    return prefixSize + 1 + uint64_t{numTotalRows_} * sizeof(PhysicalType);
  }
  if (selectedEncoding == EncodingType::Trivial ||
      selectedEncoding == EncodingType::FixedBitWidth ||
      selectedEncoding == EncodingType::Varint) {
    *size = *size - EncodingPrefix::kFixedPrefixSize + prefixSize;
    if (selectedEncoding == EncodingType::FixedBitWidth) {
      *size += FixedBitArray::bufferSize(0, 0);
    }
  }
  return *size;
}

template <typename T>
std::optional<uint64_t> SampledCost<T>::estimateSize(
    EncodingType encodingType,
    EncodingSelectionPolicyBase* policy) {
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALPRD) {
      return ALPRDEncodingBase::estimateSize(
          sampleValues_, numTotalRows_, options_, policy);
    }
    if (policy != nullptr) {
      if (auto size = estimateNestedFloatingPointSize<T>(
              encodingType,
              sampleValues_,
              numTotalRows_,
              statistics_,
              *policy,
              options_)) {
        return size;
      }
    }
  }
  if (numTotalRows_ == sampleValues_.size()) {
    return detail::EncodingSizeEstimation<T>::estimateSize(
        encodingType, sampleValues_, statistics_, options_);
  }
  NIMBLE_CHECK(
      !sampleValues_.empty(), "Size estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(sampleValues_.size(), numTotalRows_);
  const auto prefixSize =
      EncodingPrefix::serializedSize(numTotalRows_, options_.useVarintRowCount);
  const auto samplePrefixSize = EncodingPrefix::serializedSize(
      sampleValues_.size(), options_.useVarintRowCount);
  if (encodingType == EncodingType::Constant) {
    auto size = detail::EncodingSizeEstimation<T>::estimateSize(
        encodingType, sampleValues_, statistics_, options_);
    return size ? std::optional<uint64_t>{*size - samplePrefixSize + prefixSize}
                : std::nullopt;
  }
  if constexpr (!isStringType<T>()) {
    if (encodingType == EncodingType::Trivial ||
        encodingType == EncodingType::FixedBitWidth ||
        encodingType == EncodingType::SimdForBitpack) {
      return detail::EncodingSizeEstimation<T>::estimateSize(
          encodingType, numTotalRows_, statistics_, options_);
    }
  }
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALP) {
      return ALPEncoding<T>::estimateSizeFromSample(
          numTotalRows_, sampleValues_, options_);
    }
  }
  auto size = detail::EncodingSizeEstimation<T>::estimateSize(
      encodingType, sampleValues_, statistics_, options_);
  if (!size) {
    return std::nullopt;
  }
  // Project the sampled bytes after the outer prefix to the full row count:
  //
  //   estimatedSize = fullPrefixSize
  //       + (sampleSizeBytes - samplePrefixSize)
  //           * numTotalRows / numSampleRows
  //
  // Here numSampleRows is sampleValues.size(). Count the outer prefix once;
  // its varint length can depend on the row count.
  //
  // Existing composite estimates are heuristics. Scaling their inner metadata
  // along with the payload is conservative; it avoids assuming a different
  // child codec merely because the sample is small.
  //
  // Varint's existing estimator uses a fixed prefix. Keep that convention for
  // policy scoring, then correct it for the selected child's serialized size.
  const auto estimatedPrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : prefixSize;
  const auto estimatedSamplePrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : samplePrefixSize;
  return estimatedPrefixSize +
      (*size - std::min<uint64_t>(*size, estimatedSamplePrefixSize)) *
      numTotalRows_ / sampleValues_.size();
}

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
    const auto index = sampledRowIndex(i, numSamples_, values.size());
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
        SampledCost<uint16_t>{
            *childPolicies_.codes,
            {codes_.data(), numSamples_},
            numRows_,
            options_}
            .selectedSize(),
        SampledCost<PhysicalType>{
            *childPolicies_.rightParts,
            {rightParts_.data(), numSamples_},
            numRows_,
            options_}
            .selectedSize(),
        numExceptions == 0
            ? 0
            : SampledCost<uint32_t>{
                  *childPolicies_.exceptionPositions,
                  {exceptionPositions_.data(), sampleExceptions},
                  numExceptions,
                  options_}
                  .selectedSize(),
        numExceptions == 0
            ? 0
            : SampledCost<uint16_t>{
                  *childPolicies_.exceptionHighParts,
                  {exceptionHighParts_.data(), sampleExceptions},
                  numExceptions,
                  options_}
                  .selectedSize(),
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
    uint32_t numTotalRows,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  if (sampleValues.empty()) {
    return std::nullopt;
  }
  return trainSplit(sampleValues, numTotalRows, options, policy).size;
}

template <typename T>
std::optional<uint64_t> ALPRDEncodingBase::estimateNestedSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase& policy) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return estimateNestedFloatingPointSize<T>(
      encodingType, values, values.size(), statistics, policy, options);
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

template std::optional<uint64_t> ALPRDEncodingBase::estimateNestedSize<float>(
    EncodingType,
    std::span<const uint32_t>,
    const Statistics<uint32_t>&,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template std::optional<uint64_t> ALPRDEncodingBase::estimateNestedSize<double>(
    EncodingType,
    std::span<const uint64_t>,
    const Statistics<uint64_t>&,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);

} // namespace facebook::nimble
