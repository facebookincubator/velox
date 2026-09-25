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

#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/encodings/BitRangeSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"

namespace facebook::nimble {

using EncodingSelectionPolicyCreator =
    std::function<std::unique_ptr<EncodingSelectionPolicyBase>(DataType)>;

// The following enables encoding selection debug messages. By default, these
// logs are turned off (with zero overhead). In tests (or in debug sessions), we
// enable these logs.
#ifndef NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS
#define NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS 50
#endif

#ifdef NIMBLE_ENCODING_SELECTION_DEBUG
#define NIMBLE_SELECTION_LOG(stream) LOG(INFO) << stream
#else
#define NIMBLE_SELECTION_LOG(stream)
#endif

#define COMMA ,
#define UNIQUE_PTR_FACTORY_EXTRA(data_type, class, extra_types, ...)     \
  switch (data_type) {                                                   \
    case facebook::nimble::DataType::Uint8: {                            \
      return std::make_unique<class<uint8_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Int8: {                             \
      return std::make_unique<class<int8_t extra_types>>(__VA_ARGS__);   \
    }                                                                    \
    case facebook::nimble::DataType::Uint16: {                           \
      return std::make_unique<class<uint16_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int16: {                            \
      return std::make_unique<class<int16_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Uint32: {                           \
      return std::make_unique<class<uint32_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int32: {                            \
      return std::make_unique<class<int32_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Uint64: {                           \
      return std::make_unique<class<uint64_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int64: {                            \
      return std::make_unique<class<int64_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Float: {                            \
      return std::make_unique<class<float extra_types>>(__VA_ARGS__);    \
    }                                                                    \
    case facebook::nimble::DataType::Double: {                           \
      return std::make_unique<class<double extra_types>>(__VA_ARGS__);   \
    }                                                                    \
    case facebook::nimble::DataType::Bool: {                             \
      return std::make_unique<class<bool extra_types>>(__VA_ARGS__);     \
    }                                                                    \
    case facebook::nimble::DataType::String: {                           \
      return std::make_unique<class<std::string_view extra_types>>(      \
          __VA_ARGS__);                                                  \
    }                                                                    \
    case facebook::nimble::DataType::Undefined:                          \
      break;                                                             \
  }                                                                      \
  NIMBLE_UNREACHABLE("Unsupported data type {}.", toString(data_type))

#define UNIQUE_PTR_FACTORY(data_type, class, ...) \
  UNIQUE_PTR_FACTORY_EXTRA(data_type, class, , __VA_ARGS__)

/// Whether a nested stream is decoded once when its parent encoding is
/// constructed, rather than on every read that touches it. Only
/// FrequencyPartition's tag stream qualifies today.
inline bool isDecodedOnceAtConstruction(
    EncodingType parentEncodingType,
    std::optional<NestedEncodingIdentifier> nestedEncodingIdentifier) {
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  return parentEncodingType == EncodingType::FrequencyPartition &&
      nestedEncodingIdentifier.has_value() &&
      nestedEncodingIdentifier.value() ==
      EncodingIdentifiers::FrequencyPartition::TierTags;
#else
  (void)parentEncodingType;
  (void)nestedEncodingIdentifier;
  return false;
#endif
}

/// The encodings a nested stream may be chosen from, given the candidates its
/// parent was chosen from and the encoding the parent settled on. A free
/// function rather than a method because both ManualEncodingSelectionPolicy
/// below and the benchmark policy in SubstreamCompression.h must reach the
/// same answer; a second, divergent copy is hard to notice since both sides
/// still produce plausible sizes.
inline std::vector<std::pair<EncodingType, float>> nestedEncodingReadFactors(
    const std::vector<std::pair<EncodingType, float>>& parentReadFactors,
    EncodingType parentEncodingType,
    std::optional<NestedEncodingIdentifier> nestedEncodingIdentifier =
        std::nullopt) {
  std::vector<std::pair<EncodingType, float>> nested;
  nested.reserve(parentReadFactors.size());
  // Excludes encodings already selected in parent levels: not strictly
  // required, but guarantees convergence and speeds up nested selection.
  for (const auto& entry : parentReadFactors) {
    // BitRangeSplit sections are restricted to encodings that can decode one
    // of its bit ranges; every other parent only excludes itself.
    const bool isCandidate = parentEncodingType == EncodingType::BitRangeSplit
        ? detail::BitRangeSplitEncodingBase::isValidSectionEncodingCandidate(
              entry.first)
        : entry.first != parentEncodingType;
    if (isCandidate) {
      nested.emplace_back(entry);
    }
  }
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  // SubIntSplit decomposes its input into bit-range segments, each
  // independently re-encoded via encodeNested(); segments often look very
  // different from the original column, so extra integer-compression
  // candidates are offered here beyond the global default read factors.
  // This list reaches the whole subtree, not only direct children of a
  // SubIntSplit node: recursion is bounded because each candidate's own
  // encoding type (not SubIntSplit) is what gets passed down to its
  // children, so it drops itself out at the next level.
  if (parentEncodingType == EncodingType::SubIntSplit) {
    for (const auto& pair :
         {// PFOR, SimdForBitpack and BlockBitPacking are held above the
          // delta family's factor; levelling them cedes bulk decode
          // throughput and point latency to a small compression gain. Do
          // not change without re-measuring.
          std::pair{EncodingType::PFOR, 0.9f},
          std::pair{EncodingType::SimdForBitpack, 0.9f},
          std::pair{EncodingType::BlockBitPacking, 0.9f},
          std::pair{EncodingType::Delta, 0.85f},
          std::pair{EncodingType::FOR, 0.85f},
          // Huffman is deliberately absent: it decodes bit-serially, which
          // costs bulk decode throughput here. See
          // Encoding::Options::subIntSplitAllowHuffman for the opt-in.
          //
          // DeltaBlock is deliberately absent too: its serial prefix-sum
          // decode does not vectorize, and its per-block baselines (which
          // pay off on a gather or low-selectivity read) are untested here.
          std::pair{EncodingType::FrequencyPartition, 0.85f}}) {
      nested.push_back(pair);
    }
  }
#endif

  // Streams decoded once at construction, rather than on every read, can
  // admit encodings that are rightly withheld from section payloads (where
  // decode cost is per-read): the reader works from the already-decoded
  // form, so the decode cost is paid once per encoding, not once per row.
  // Huffman is the case in point, admitted here despite being withheld from
  // the SubIntSplit list above. Membership is by role, not by encoding: add
  // any future once-at-construction stream here.
  if (isDecodedOnceAtConstruction(
          parentEncodingType, nestedEncodingIdentifier)) {
    if (std::find_if(nested.begin(), nested.end(), [](const auto& entry) {
          return entry.first == EncodingType::Huffman;
        }) == nested.end()) {
      nested.emplace_back(EncodingType::Huffman, 0.85f);
    }
  }

  return nested;
}

/// The read factor select() weighs `encodingType` by: the table's factor,
/// except Trivial's is withheld (raised to 1.0) where taking it would cost
/// compression. Trivial stores at the storage type's width rather than the
/// value width, so against a stream it does not fit exactly it is always
/// larger; its low factor can still let it out-cost a narrower FixedBitWidth
/// on the weighted comparison, silently spending bits on decode speed.
/// Withholding it only when it is no smaller than FixedBitWidth keeps the
/// discount wherever it is actually free, and does not by itself hand the
/// stream to FixedBitWidth if some other candidate's own factor beats it.
///
/// A free function because anything modelling selection, such as the oracle
/// harness, must reach the same answer or silently drift from it.
inline float effectiveReadFactor(
    EncodingType encodingType,
    float tableReadFactor,
    uint64_t estimatedSize,
    const std::optional<uint64_t>& fixedBitWidthSize) {
  const bool trivialKeepsItsDiscount = encodingType != EncodingType::Trivial ||
      !fixedBitWidthSize.has_value() || estimatedSize <= *fixedBitWidthSize;
  return trivialKeepsItsDiscount ? tableReadFactor : 1.0f;
}

/// Whether `encodingType` is certain to cost at least `minCost`, from a
/// lower bound on its estimate, so selection may skip estimating it without
/// changing the result. Trivial is never skipped, since its read factor
/// depends on its own estimate.
template <typename T>
bool candidateCannotWin(
    EncodingType encodingType,
    float readFactor,
    double minCost,
    std::span<const typename TypeTraits<T>::physicalType> values,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    const Encoding::Options& options) {
  if (encodingType == EncodingType::Trivial ||
      minCost == std::numeric_limits<double>::max()) {
    return false;
  }
  const auto lowerBound =
      detail::EncodingSizeEstimation<T>::estimateSizeLowerBound(
          encodingType, values, statistics, options);
  // The same expression select() costs an estimate with, so that a bound equal
  // to the estimate rounds to the same cost.
  return lowerBound.has_value() &&
      static_cast<double>(lowerBound.value() * readFactor) >= minCost;
}

/// Whether selection's screen may withhold `encodingType` from pricing on the
/// whole stream. These are the candidates that price a stream from its
/// distinct values or its runs; see Encoding::Options::selectionScreenRows.
inline bool isScreenedBySample(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::MainlyConstant:
    case EncodingType::Dictionary:
    case EncodingType::RLE:
    case EncodingType::Huffman:
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    case EncodingType::FrequencyPartition:
#endif
      return true;
    default:
      return false;
  }
}

// Numeric body of screenCandidatesBySample.
template <typename T>
void screenNumericCandidatesBySample(
    std::span<const typename TypeTraits<T>::physicalType> values,
    std::vector<std::pair<EncodingType, float>>& candidates,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  const size_t sampleRows = options.selectionScreenRows;
  if (sampleRows == 0 || values.size() <= 2 * sampleRows ||
      std::none_of(candidates.begin(), candidates.end(), [](const auto& entry) {
        return isScreenedBySample(entry.first);
      })) {
    return;
  }

  constexpr size_t kBlocks{8};
  const auto sample = sampleSpreadBlocks(
      values, kBlocks, std::max<size_t>(sampleRows / kBlocks, 1));
  const std::span<const physicalType> sampleValues{sample};
  const auto sampleStatistics = Statistics<physicalType>::create(sampleValues);

  std::optional<uint64_t> fixedBitWidthSize;
  if (std::any_of(candidates.begin(), candidates.end(), [](const auto& entry) {
        return entry.first == EncodingType::FixedBitWidth;
      })) {
    fixedBitWidthSize = detail::EncodingSizeEstimation<T>::estimateSize(
        EncodingType::FixedBitWidth, sampleValues, sampleStatistics, options);
  }
  std::vector<std::optional<double>> sampleCosts;
  sampleCosts.reserve(candidates.size());
  double cheapest = std::numeric_limits<double>::max();
  for (const auto& [encodingType, readFactor] : candidates) {
    const auto estimatedSize = detail::EncodingSizeEstimation<T>::estimateSize(
        encodingType, sampleValues, sampleStatistics, options);
    if (!estimatedSize.has_value()) {
      sampleCosts.emplace_back();
      continue;
    }
    const double cost =
        static_cast<double>(estimatedSize.value()) *
        effectiveReadFactor(
            encodingType, readFactor, estimatedSize.value(), fixedBitWidthSize);
    sampleCosts.emplace_back(cost);
    cheapest = std::min(cheapest, cost);
  }

  const double bound = cheapest * options.selectionScreenMargin;
  size_t kept{0};
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!isScreenedBySample(candidates[i].first) ||
        !sampleCosts[i].has_value() || sampleCosts[i].value() <= bound) {
      candidates[kept++] = candidates[i];
    }
  }
  candidates.resize(kept);
}

/// Drops from `candidates` each costly candidate whose price on a sample of
/// `values` exceeds the cheapest candidate's sample price by more than
/// Encoding::Options::selectionScreenMargin. Leaves `candidates` untouched
/// when the screen is off or the stream is too short to be worth sampling.
/// A candidate the sample could not price at all is kept.
template <typename T>
void screenCandidatesBySample(
    std::span<const typename TypeTraits<T>::physicalType> values,
    std::vector<std::pair<EncodingType, float>>& candidates,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  // Numbers only, which is what the screen has been measured on. Booleans
  // cannot be sampled into a span at all.
  if constexpr (!isNumericType<physicalType>() || isBoolType<physicalType>()) {
    return;
  } else {
    screenNumericCandidatesBySample<T>(values, candidates, options);
  }
}

/// Manual encoding selection implementation.
/// Uses a manually crafted model to choose the most appropriate encoding based
/// on the provided statistics.
template <typename T>
class ManualEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  ManualEncodingSelectionPolicy(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors,
      std::optional<CompressionOptions> compressionOptions,
      std::optional<NestedEncodingIdentifier> identifier,
      std::optional<std::vector<std::pair<EncodingType, float>>>
          nestedEncodingReadFactors = std::nullopt)
      : candidateEncodingReadFactors_{std::move(encodingReadFactors)},
        compressionOptions_{std::move(compressionOptions)},
        identifier_{identifier},
        nestedEncodingReadFactorsOverride_{
            std::move(nestedEncodingReadFactors)} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    if (values.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    auto candidateEncodingReadFactors = candidateEncodingReadFactors_;
    // TODO: Remove this opt-in once ALP is production-ready for default
    // selection.
    if constexpr (isFloatingPointType<T>()) {
      if (options.allowNestedAlpSelection &&
          (identifier_ == EncodingIdentifiers::Dictionary::Alphabet ||
           identifier_ == EncodingIdentifiers::MainlyConstant::OtherValues ||
           identifier_ == EncodingIdentifiers::RunLength::RunValues) &&
          std::none_of(
              candidateEncodingReadFactors.begin(),
              candidateEncodingReadFactors.end(),
              [](const auto& entry) {
                return entry.first == EncodingType::ALP;
              })) {
        candidateEncodingReadFactors.emplace_back(EncodingType::ALP, 1.0);
      }
    }

    // A nested stream (one this policy was created for by a parent encoding)
    // is not offered SubIntSplit when the caller withholds it; see
    // Encoding::Options::subIntSplitInNestedStreams.
    if (identifier_.has_value() && !options.subIntSplitInNestedStreams) {
      candidateEncodingReadFactors.erase(
          std::remove_if(
              candidateEncodingReadFactors.begin(),
              candidateEncodingReadFactors.end(),
              [](const auto& entry) {
                return entry.first == EncodingType::SubIntSplit;
              }),
          candidateEncodingReadFactors.end());
    }

    // A bit-flip admission decides from the profile alone whether SubIntSplit
    // is worth costing: a rejected stream loses the candidate, an admitted
    // one still has to win the ordinary size comparison.
    // Options::subIntSplitAdmissionForces instead leaves an admitted stream
    // with SubIntSplit as its only candidate.
    bool subIntSplitForced{false};
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    if constexpr (
        isIntegralType<T>() &&
        (sizeof(physicalType) == 4 || sizeof(physicalType) == 8)) {
      const auto admission = static_cast<subintsplit::SubIntSplitAdmission>(
          options.subIntSplitAdmission);
      const auto subIntSplit = std::find_if(
          candidateEncodingReadFactors.begin(),
          candidateEncodingReadFactors.end(),
          [](const auto& entry) {
            return entry.first == EncodingType::SubIntSplit;
          });
      if (admission != subintsplit::SubIntSplitAdmission::kEstimate &&
          subIntSplit != candidateEncodingReadFactors.end()) {
        const bool admitted = options.subIntSplitAdmissionProfilePairs == 0
            ? subintsplit::bitFlipAdmits(
                  statistics.bitFlipProfile(),
                  admission,
                  subintsplit::TopLevelPolicyConfig{})
            : subintsplit::bitFlipAdmits(
                  subintsplit::bitFlipAdmissionProfile(
                      values,
                      admission,
                      options.subIntSplitAdmissionProfilePairs),
                  admission,
                  subintsplit::TopLevelPolicyConfig{});
        if (!admitted) {
          candidateEncodingReadFactors.erase(subIntSplit);
        } else if (options.subIntSplitAdmissionForces) {
          const auto entry = *subIntSplit;
          candidateEncodingReadFactors.assign(1, entry);
          subIntSplitForced = true;
        }
      }
    }
#endif

    // Not while decode is priced: the screen compares sizes, and would drop a
    // candidate that loses on size and wins once its decode is counted.
    if (!subIntSplitForced &&
        (!options.subIntSplitSectionSelection ||
         options.subIntSplitDecodeWeight == 0.0)) {
      screenCandidatesBySample<T>(
          values, candidateEncodingReadFactors, options);
    }

    // Fast path: when there are no candidate encodings, fall back to Trivial.
    if (candidateEncodingReadFactors.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    // A size estimate counts bytes on disk, which are compressed bytes when
    // this policy hands its streams to a substream compressor; this flag
    // tells an estimate which world it is pricing. Copied, not mutated,
    // since the caller's options are shared with the encode.
    Encoding::Options estimationOptions = options;
    estimationOptions.substreamCompression = compressionOptions_.has_value() &&
        compressionOptions_->compressionType != CompressionType::Uncompressed;

    // FixedBitWidth's size, when it is a candidate, so effectiveReadFactor
    // can withhold Trivial's discount where taking it would cost compression.
    std::optional<uint64_t> fixedBitWidthSize;
    if (std::any_of(
            candidateEncodingReadFactors.begin(),
            candidateEncodingReadFactors.end(),
            [](const auto& entry) {
              return entry.first == EncodingType::FixedBitWidth;
            })) {
      fixedBitWidthSize = detail::EncodingSizeEstimation<T>::estimateSize(
          EncodingType::FixedBitWidth, values, statistics, estimationOptions);
    }

    // How much a section's decode counts against its size, and for which
    // read shape. Zero unless this is a SubIntSplit section and the caller
    // asked for decode to count, in which case the term below vanishes and
    // selection falls back to size alone.
    const double decodeWeight = options.subIntSplitSectionSelection
        ? options.subIntSplitDecodeWeight
        : 0.0;
    const auto decodePattern = static_cast<subintsplit::DecodeAccessPattern>(
        options.subIntSplitDecodeAccessPattern);

    // Costs are compared in double so the decode term, which is in bytes and
    // can be large, does not lose the size term to rounding.
    double minCost = std::numeric_limits<double>::max();
    EncodingType selectedEncoding = EncodingType::Trivial;
    std::optional<uint64_t> selectedEstimatedSize;
    // What size alone would have chosen, held against the decode-weighted
    // winner below. Tracked unconditionally, since it is the incumbent when
    // the weight is zero.
    double minSizeCost = std::numeric_limits<double>::max();
    EncodingType sizeSelectedEncoding = EncodingType::Trivial;
    std::optional<uint64_t> sizeSelectedEstimatedSize;
    // Iterate on all candidate encodings, and pick the encoding with the
    // minimal cost.
    for (const auto& entry : candidateEncodingReadFactors) {
      const auto encodingType = entry.first;
      if (decodeWeight == 0.0 &&
          candidateCannotWin<T>(
              encodingType,
              entry.second,
              minCost,
              values,
              statistics,
              estimationOptions)) {
        continue;
      }
      const auto estimatedSize =
          detail::EncodingSizeEstimation<T>::estimateSize(
              encodingType, values, statistics, estimationOptions);
      if (!estimatedSize.has_value()) {
        NIMBLE_SELECTION_LOG(encodingType << " encoding is incompatible.");
        continue;
      }

      // Read factor weights raise/lower the favorability of each encoding,
      // except where Trivial's would be unearned; see effectiveReadFactor.
      const auto readFactor = effectiveReadFactor(
          encodingType, entry.second, estimatedSize.value(), fixedBitWidthSize);
      // Size, plus what reading the section back costs. Section decode times
      // add rather than max, so a slow section is paid in full by every scan
      // of the column, which choosing on size alone cannot see. See
      // subintsplit/DecodeCost.h for the per-encoding rates.
      const double sizeCost =
          static_cast<double>(estimatedSize.value() * readFactor);
      if (sizeCost < minSizeCost) {
        minSizeCost = sizeCost;
        sizeSelectedEncoding = encodingType;
        sizeSelectedEstimatedSize = estimatedSize;
      }
      double cost = sizeCost;
      if (decodeWeight != 0.0) {
        const double nanosPerRow = subintsplit::decodeNanosPerRow(
            encodingType,
            decodePattern,
            static_cast<subintsplit::DecodeReadPath>(
                options.subIntSplitDecodeReadPath),
            static_cast<double>(estimatedSize.value()) * 8.0,
            values.size());
        cost += subintsplit::decodeCostBits(
                    nanosPerRow, values.size(), decodeWeight) /
            8.0;
      }
      NIMBLE_SELECTION_LOG(
          "Encoding: " << encodingType << ", Size: "
                       << velox::succinctBytes(estimatedSize.value())
                       << ", Factor: " << readFactor << ", Cost: " << cost);
      if (cost < minCost) {
        minCost = cost;
        selectedEncoding = encodingType;
        selectedEstimatedSize = estimatedSize;
      }
    }

    // Bounded on size: without this, a section can be handed an encoding
    // that reads faster but stores the column arbitrarily worse.
    if (decodeWeight != 0.0 && selectedEstimatedSize.has_value() &&
        sizeSelectedEstimatedSize.has_value()) {
      const double allowedSize =
          static_cast<double>(sizeSelectedEstimatedSize.value()) *
          (1.0 + options.subIntSplitMaxSizeRegression);
      if (static_cast<double>(selectedEstimatedSize.value()) > allowedSize) {
        selectedEncoding = sizeSelectedEncoding;
        selectedEstimatedSize = sizeSelectedEstimatedSize;
      }
    }

    NIMBLE_SELECTION_LOG(
        "Selected Encoding"
        << (identifier_.has_value()
                ? folly::to<std::string>(
                      " [NestedEncodingIdentifier: ", identifier_.value(), "]")
                : "")
        << ": " << selectedEncoding << ", Sampled Data: "
        << folly::join(
               ",",
               std::span<const physicalType>{
                   values.data(),
                   std::min(
                       size_t(NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS),
                       values.size())})
        << (values.size() > size_t(NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS)
                ? "..."
                : ""));
    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = selectedEncoding,
          .encodingConfig = {},
          .estimatedSize = selectedEstimatedSize};
    }
    // Encoding selection optimizes the in-memory layout. Compression is still
    // attempted for leaf data streams to reduce persistent storage size.
    return {
        .encodingType = selectedEncoding,
        .encodingConfig = {},
        .estimatedSize = selectedEstimatedSize,
        .compressionPolicyFactory = [compressionOptions =
                                         compressionOptions_.value(),
                                     selectedEncoding]() {
          return std::make_unique<ConfiguredCompressionPolicy>(
              compressionOptions, selectedEncoding);
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {
        .encodingType = EncodingType::Nullable,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

  const std::vector<std::pair<EncodingType, float>>&
  candidateEncodingReadFactors() const {
    return candidateEncodingReadFactors_;
  }

  std::unique_ptr<EncodingSelectionPolicy<T>> narrowed(
      const std::function<bool(EncodingType)>& keep) const override {
    std::vector<std::pair<EncodingType, float>> kept;
    for (const auto& entry : candidateEncodingReadFactors_) {
      if (keep(entry.first)) {
        kept.push_back(entry);
      }
    }
    // Nested streams are offered what createImpl would offer them.
    return std::make_unique<ManualEncodingSelectionPolicy<T>>(
        std::move(kept),
        compressionOptions_,
        identifier_,
        nestedEncodingReadFactorsOverride_.has_value()
            ? nestedEncodingReadFactorsOverride_.value()
            : candidateEncodingReadFactors_);
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // The candidate list is decided by nestedEncodingReadFactors above, not
    // here, so that a benchmark policy standing in for this one reaches the
    // same list through the same code rather than through a copy of it.
    const auto& sourceEncodingReadFactors =
        nestedEncodingReadFactorsOverride_.has_value()
        ? nestedEncodingReadFactorsOverride_.value()
        : candidateEncodingReadFactors_;
    auto nestedEncodingReadFactors = nimble::nestedEncodingReadFactors(
        sourceEncodingReadFactors,
        parentEncodingType,
        nestedEncodingIdentifier);
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        compressionOptions_,
        nestedEncodingIdentifier,
        std::nullopt);
  }

 private:
  // Candidate encodings and their read-cost factors. Encoding selection uses
  // estimatedSize * readFactor as the cost, so a lower factor makes an
  // encoding more likely to be picked. Right now, these represent mostly the
  // CPU cost to decode values. Trivial is boosted with a lower factor because
  // it also benefits from applying compression.
  // See ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors for
  // the default.
  const std::vector<std::pair<EncodingType, float>>
      candidateEncodingReadFactors_;
  // When nullopt, compression is disabled for this policy and all nested
  // policies created from it.
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<NestedEncodingIdentifier> identifier_;
  // An override for the next nested policy. Deeper policies inherit their
  // parent's already-filtered candidates.
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactorsOverride_;
};

class ManualEncodingSelectionPolicyFactory {
 public:
  /// TODO: Add ALP once it is production-ready for default manual selection.
  static std::vector<std::pair<EncodingType, float>>
  defaultEncodingReadFactors();

  /// Parses semicolon-delimited runtime read-factor config.
  /// Allows explicit opt-in to parseable encodings that are not part of
  /// defaultEncodingReadFactors().
  static std::vector<std::pair<nimble::EncodingType, float>>
  parseEncodingReadFactors(const std::string& readFactorsConfig);

  /// Builds a factory from a nimble.encoding_selection_config string of the
  /// form "type:default[,read_factors:<E1>=<f1>;<E2>=<f2>;...]". Ignores the
  /// 'type' key (already used by createEncodingSelectionPolicyFactory). Returns
  /// nullopt when no 'read_factors' are given, so the caller keeps the
  /// nimble.manual_encoding_selection_read_factors default; otherwise a factory
  /// whose read factors override that config. NimbleUserError on a malformed
  /// entry or unknown key.
  static std::optional<ManualEncodingSelectionPolicyFactory> create(
      std::string_view configStr,
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{});

  ManualEncodingSelectionPolicyFactory(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors =
          defaultEncodingReadFactors(),
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{},
      std::optional<std::vector<std::pair<EncodingType, float>>>
          nestedEncodingReadFactors = std::nullopt);

  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const;

  /// All encodings the string-config parsers accept (production +
  /// experimental). Also used to resolve encoding names to EncodingType.
  static std::vector<EncodingType> possibleEncodings();

 private:
  const std::vector<std::pair<EncodingType, float>> encodingReadFactors_;
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactors_;
};

/// Learned encoding selection implementation.
template <typename T>
class LearnedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  /// Model trained offline for learned encoding selection.
  /// Parameters are relatively robust and do not need updates unless encodings
  /// are added or removed.
  struct EncodingPredictionModel {
    explicit EncodingPredictionModel()
        : maxRepeatParam(1.52), minRepeatParam(1.13), uniqueParam(2.589) {}

    float maxRepeatParam;
    float minRepeatParam;
    float uniqueParam;

    float predict(const Statistics<physicalType>& statistics) {
      // TODO: Utilize more features within statistics for prediction.
      const auto maxRepeat = statistics.maxRepeat();
      const auto minRepeat = statistics.minRepeat();
      const auto unique = statistics.uniqueCounts().value().size();
      return maxRepeatParam * maxRepeat + minRepeatParam * minRepeat +
          uniqueParam * unique;
    }
  };

  LearnedEncodingSelectionPolicy(
      std::vector<EncodingType> encodingChoices,
      std::optional<NestedEncodingIdentifier> identifier,
      EncodingPredictionModel encodingModel = EncodingPredictionModel())
      : encodingChoices_{std::move(encodingChoices)},
        identifier_{identifier},
        mlModel_{encodingModel} {}

  LearnedEncodingSelectionPolicy()
      : LearnedEncodingSelectionPolicy{
            possibleEncodingChoices(),
            std::nullopt,
            EncodingPredictionModel()} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override;

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {
        .encodingType = EncodingType::Nullable,
    };
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // In each sub-level of the encoding selection, we exclude the encodings
    // selected in parent levels. Although this is not required (as hopefully,
    // the model will not pick a nested encoding of the same type as the
    // parent), it provides an additional safety net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    //
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<EncodingType> nestedEncodingChoices;
    nestedEncodingChoices.reserve(encodingChoices_.size());
    for (const auto& encodingType_ : encodingChoices_) {
      if (encodingType_ != parentEncodingType) {
        nestedEncodingChoices.push_back(encodingType_);
      }
    }
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        LearnedEncodingSelectionPolicy,
        std::move(nestedEncodingChoices),
        nestedEncodingIdentifier);
  }

 private:
  // TODO: Add ALP once it is production-ready for learned selection.
  static std::vector<EncodingType> possibleEncodingChoices() {
    return {
        EncodingType::Constant,
        EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::MainlyConstant,
        EncodingType::SparseBool,
        EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::Varint,
    };
  }

  std::vector<EncodingType> encodingChoices_;
  std::optional<NestedEncodingIdentifier> identifier_;
  EncodingPredictionModel mlModel_;
};

template <typename T>
EncodingSelectionResult LearnedEncodingSelectionPolicy<T>::select(
    std::span<const typename LearnedEncodingSelectionPolicy<T>::physicalType>
        values,
    const Statistics<typename LearnedEncodingSelectionPolicy<T>::physicalType>&
        statistics,
    const Encoding::Options& /* options */) {
  if (values.empty()) {
    return {
        .encodingType = EncodingType::Trivial,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

  const auto prediction = mlModel_.predict(statistics);
  if (prediction > 0.1) {
    return {
        .encodingType = EncodingType::Trivial,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }
  // TODO: Implement a multi-class Encoding model so that we can predict not
  // only a trivial encoding but also other encodings.

  return {
      .encodingType = EncodingType::Trivial,
      .encodingConfig = {},
      .estimatedSize = std::nullopt,
  };
}

template <typename T>
class ReplayedEncodingSelectionPolicy
    : public nimble::EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  ReplayedEncodingSelectionPolicy(
      EncodingLayout encodingLayout,
      std::optional<CompressionOptions> compressionOptions,
      const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator)
      : compressionOptions_{std::move(compressionOptions)},
        encodingSelectionPolicyCreator_{encodingSelectionPolicyCreator},
        encodingLayout_{std::move(encodingLayout)} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = encodingLayout_.encodingType(),
          .encodingConfig = encodingLayout_.config(),
          .estimatedSize = std::nullopt,
      };
    }
    return {
        .encodingType = encodingLayout_.encodingType(),
        .encodingConfig = encodingLayout_.config(),
        .estimatedSize = std::nullopt,
        .compressionPolicyFactory = [this]() {
          return std::make_unique<ReplayedCompressionPolicy>(
              encodingLayout_.compressionType(), compressionOptions_.value());
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    // NullableEncoding asks createImpl() for nullable data and nulls children.
    // The replay policy is initialized with the data layout, so synthesize the
    // nullable parent shape here.
    encodingLayout_ = EncodingLayout{
        EncodingType::Nullable,
        {},
        CompressionType::Uncompressed,
        {
            /*Data=*/std::move(encodingLayout_),
            /*Nulls=*/std::nullopt,
        }};
    return {
        .encodingType = EncodingType::Nullable,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

 protected:
  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* parentEncodingType */,
      nimble::NestedEncodingIdentifier nestedEncodingIdentifier,
      nimble::DataType nestedDataType) override {
    NIMBLE_CHECK_LT(
        nestedEncodingIdentifier,
        encodingLayout_.childrenCount(),
        "Sub-encoding identifier out of range.");
    auto child = encodingLayout_.child(nestedEncodingIdentifier);

    if (child.has_value()) {
      UNIQUE_PTR_FACTORY(
          nestedDataType,
          ReplayedEncodingSelectionPolicy,
          child.value(),
          compressionOptions_,
          encodingSelectionPolicyCreator_);
    } else {
      return encodingSelectionPolicyCreator_(nestedDataType);
    }
  }

 private:
  const std::optional<CompressionOptions> compressionOptions_;
  const EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_;
  EncodingLayout encodingLayout_;
};

#undef COMMA

} // namespace facebook::nimble
