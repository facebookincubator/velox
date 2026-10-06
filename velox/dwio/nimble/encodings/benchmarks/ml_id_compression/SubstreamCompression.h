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

#include <algorithm>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gflags/gflags.h>

DECLARE_string(mlidc_sis_withdraw_nested_encodings);
DECLARE_string(mlidc_sis_allowed_encodings);
DECLARE_string(mlidc_auto_encodings);
DECLARE_uint32(mlidc_selection_screen_rows);
DECLARE_double(mlidc_selection_screen_margin);
DECLARE_uint32(mlidc_sis_section_threads);

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// Encodes benchmark data with a caller-chosen compressor applied to the
// encoding's streams, including the sub-streams of a nested encoding such as
// SubIntSplit.
//
// This exists because test::Encoder cannot express the choice. Its
// TestCompressPolicy handles only Uncompressed and Zstd, and silently
// redirects everything else to Zstd level 3 under
// DISABLE_META_INTERNAL_COMPRESSOR, which would report OpenZL numbers that are
// really Zstd. Its policy classes are private, so they cannot be reused. Its
// nested path also hard-codes ManualEncodingSelectionPolicyFactory{...,
// std::nullopt}, which leaves sub-streams on the default compressor whatever
// the caller asked for.
//
// Compressor names are parsed by nimble::toCompressionType, so a codec added
// to nimble becomes available here with no change to this file.

namespace facebook::nimble::mlidc {

// The encodings a comma-separated list names, or an empty set for an empty
// list. Each name must be one a SubIntSplit section can be offered (or
// SubIntSplit itself where admitted), so a misspelled name fails rather than
// silently holding nothing back.
inline std::unordered_set<nimble::EncodingType> parseEncodingNames(
    const std::string& names,
    std::string_view flag,
    bool admitSubIntSplit) {
  std::unordered_set<nimble::EncodingType> allowed;
  if (names.empty()) {
    return allowed;
  }
  auto candidates = nimble::nestedEncodingReadFactors(
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      nimble::EncodingType::SubIntSplit);
  if (admitSubIntSplit) {
    candidates.emplace_back(nimble::EncodingType::SubIntSplit, 1.0f);
  }
  size_t start{0};
  while (start <= names.size()) {
    const size_t end = std::min(names.find(',', start), names.size());
    const std::string name = names.substr(start, end - start);
    const auto match = std::find_if(
        candidates.begin(), candidates.end(), [&name](const auto& entry) {
          return nimble::toString(entry.first) == name;
        });
    NIMBLE_CHECK(
        match != candidates.end(),
        "An encoding list names an encoding it cannot offer: --{}={}",
        flag,
        name);
    allowed.insert(match->first);
    start = end + 1;
  }
  return allowed;
}

inline std::unordered_set<nimble::EncodingType> sisAllowedEncodings() {
  return parseEncodingNames(
      FLAGS_mlidc_sis_allowed_encodings,
      "mlidc_sis_allowed_encodings",
      /*admitSubIntSplit=*/false);
}

// A nested stream's candidates held to `allowed` and weighed by size alone,
// an empty set holding nothing back and keeping the writer's factors. A
// top-level policy keeps its own list, which may name SubIntSplit. The policy
// hands itself down, so every nested stream below it is held to the set too,
// including the extra candidates a SubIntSplit section is otherwise offered.
template <typename T>
class AllowedSetSelectionPolicy
    : public nimble::ManualEncodingSelectionPolicy<T> {
 public:
  using ReadFactors = std::vector<std::pair<nimble::EncodingType, float>>;

  AllowedSetSelectionPolicy(
      ReadFactors readFactors,
      std::unordered_set<nimble::EncodingType> allowed,
      std::optional<nimble::NestedEncodingIdentifier> identifier)
      : nimble::ManualEncodingSelectionPolicy<
            T>{identifier.has_value() ? held(readFactors, allowed) : readFactors, std::nullopt, identifier},
        readFactors_{
            identifier.has_value() ? held(std::move(readFactors), allowed)
                                   : std::move(readFactors)},
        allowed_{std::move(allowed)} {}

 protected:
  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType parentEncodingType,
      nimble::NestedEncodingIdentifier nestedEncodingIdentifier,
      nimble::DataType nestedDataType) override {
    auto nested = nimble::nestedEncodingReadFactors(
        readFactors_, parentEncodingType, nestedEncodingIdentifier);
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        AllowedSetSelectionPolicy,
        std::move(nested),
        allowed_,
        nestedEncodingIdentifier);
  }

 private:
  static ReadFactors held(
      ReadFactors readFactors,
      const std::unordered_set<nimble::EncodingType>& allowed) {
    if (allowed.empty()) {
      return readFactors;
    }
    // Weighed at 1.0 and listed once, so a restricted set is chosen from by
    // size alone at every level, even where a SubIntSplit parent would add a
    // candidate again at its own factor.
    ReadFactors held;
    for (const auto& [encodingType, readFactor] : readFactors) {
      if (allowed.count(encodingType) != 0 &&
          std::none_of(held.begin(), held.end(), [&](const auto& entry) {
            return entry.first == encodingType;
          })) {
        held.emplace_back(encodingType, 1.0f);
      }
    }
    return held;
  }

  const ReadFactors readFactors_;
  const std::unordered_set<nimble::EncodingType> allowed_;
};

// Whether E::encode takes a subintsplit::TuningConfig after its options. A
// SubIntSplit arm's planner and transform settings live only there, so a
// target encoding such an E must hand the config over or run on defaults.
template <typename E>
inline constexpr bool kEncodesWithTuning{false};

template <typename T>
inline constexpr bool kEncodesWithTuning<nimble::SubIntSplitEncoding<T>>{true};

// Answers its first select() with a result already chosen, so a caller that
// had to see selection's choice before encoding does not select twice. Every
// later select(), and every nested stream's policy, is the allowed-set
// policy's own.
template <typename T>
class PreselectedSelectionPolicy : public AllowedSetSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  using AllowedSetSelectionPolicy<T>::AllowedSetSelectionPolicy;

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> values,
      const nimble::Statistics<physicalType>& statistics,
      const nimble::Encoding::Options& options) override {
    if (preselected_.has_value()) {
      auto result = std::move(*preselected_);
      preselected_.reset();
      return result;
    }
    return AllowedSetSelectionPolicy<T>::select(values, statistics, options);
  }

  // Sets the result the next select() returns.
  void preselect(nimble::EncodingSelectionResult result) {
    preselected_ = std::move(result);
  }

 private:
  std::optional<nimble::EncodingSelectionResult> preselected_;
};

// A stream whose encoding selection chooses from --mlidc_auto_encodings, as a
// writer chooses, rather than one forced encoding. Only the view target reads
// it, since the cursor target constructs its encoding type directly.
template <typename T>
struct AutoSelectedEncoding {
  using cppDataType = T;
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  // `tuning` reaches SubIntSplit when selection picks it. EncodingFactory
  // would encode it under the default config, dropping the planner inventory
  // --mlidc_sis_allowed_encodings restricts it to, so that case is encoded
  // here directly.
  static std::string_view encode(
      nimble::EncodingSelection<physicalType>& /*selection*/,
      std::span<const physicalType> values,
      nimble::Buffer& buffer,
      const nimble::Encoding::Options& options,
      const subintsplit::TuningConfig& tuning) {
    const auto listed = parseEncodingNames(
        FLAGS_mlidc_auto_encodings,
        "mlidc_auto_encodings",
        /*admitSubIntSplit=*/true);
    // The writer's candidates when unrestricted. A listed set is weighed at
    // 1.0 throughout, so selection chooses by size alone, as BtrBlocks' and
    // FastLanes' selection does, rather than trading size for the writer's
    // decode-speed factors.
    auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
    if (!listed.empty()) {
      readFactors.clear();
      for (const auto encodingType : listed) {
        readFactors.emplace_back(encodingType, 1.0f);
      }
    }
    // Nested streams draw from the same list; SubIntSplit never nests.
    auto nestedAllowed = listed;
    nestedAllowed.erase(nimble::EncodingType::SubIntSplit);
    auto policy = std::make_unique<PreselectedSelectionPolicy<T>>(
        std::move(readFactors), std::move(nestedAllowed), std::nullopt);
    // Selected here, as EncodingFactory::encode selects, so that the choice is
    // known before anything is encoded.
    auto statistics = nimble::Statistics<physicalType>::create(values);
    auto result = policy->select(values, statistics, options);
    if constexpr (
        nimble::isNumericType<physicalType>() &&
        (sizeof(physicalType) == 4 || sizeof(physicalType) == 8)) {
      if (result.encodingType == nimble::EncodingType::SubIntSplit) {
        nimble::EncodingSelection<physicalType> selection{
            std::move(result), std::move(statistics), std::move(policy)};
        return nimble::SubIntSplitEncoding<T>::encode(
            selection, values, buffer, options, tuning);
      }
    }
    policy->preselect(std::move(result));
    std::unique_ptr<nimble::EncodingSelectionPolicy<T>> basePolicy =
        std::move(policy);
    return nimble::EncodingFactory::encode<T>(
        std::move(basePolicy),
        std::span<const T>(
            reinterpret_cast<const T*>(values.data()), values.size()),
        buffer,
        options);
  }

  // Decodes whatever encoding selection chose; only --mlidc_dump_encoding
  // builds one, to report the encoding tree.
  template <typename StringBufferFactory>
  AutoSelectedEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      StringBufferFactory stringBufferFactory,
      const nimble::Encoding::Options& options)
      : encoding_{nimble::EncodingFactory{options}.create(
            pool,
            data,
            std::move(stringBufferFactory),
            options)} {}

  std::string debugString(int offset) const {
    return encoding_->debugString(offset);
  }

 private:
  std::unique_ptr<nimble::Encoding> encoding_;
};

template <typename T>
inline constexpr bool kEncodesWithTuning<AutoSelectedEncoding<T>>{true};

// Rejects compressors that have no OSS implementation, rather than quietly
// substituting a different one.
inline CompressionType parseCompressionType(const std::string& name) {
  auto type = nimble::toCompressionType(name);
#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  if (type == CompressionType::MetaInternal) {
    throw std::runtime_error(
        "MetaInternal has no OSS implementation and is not in the compressor "
        "registry. Use Uncompressed, Zstd, Lz4 or OpenZL.");
  }
#endif
  return type;
}

// Builds the per-stream compression config for one chosen compressor. Unlike
// test::Encoder's policy this never substitutes a different compressor.
class BenchCompressPolicy : public nimble::CompressionPolicy {
 public:
  explicit BenchCompressPolicy(CompressionType compressionType)
      : compressionType_{compressionType} {}

  nimble::CompressionConfig config() const override {
    nimble::CompressionConfig config{.compressionType = compressionType_};
    // Defaults mirror CompressionOptions so a stream compressed here matches
    // one compressed through the production selection path.
    config.parameters.zstd.compressionLevel = 3;
    return config;
  }

  bool shouldAccept(
      nimble::CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return true;
  }

 private:
  CompressionType compressionType_;
};

// Returns the CompressionOptions handed to nested selection, so sub-streams
// use the same compressor as the top-level stream.
inline CompressionOptions compressionOptionsFor(CompressionType type) {
  CompressionOptions options;
  options.compressionType = type;
  // The default 0.98 accept ratio would silently leave a stream uncompressed
  // when the codec barely helps, which makes a compressor comparison read as a
  // codec difference. Always keep the requested compressor's output.
  options.compressionAcceptRatio = 1.0f;
  // Nimble skips compression for streams below these sizes. Benchmarks want
  // the requested compressor applied uniformly.
  options.zstdMinCompressionSize = 0;
  options.lz4MinCompressionSize = 0;
  options.openzlMinCompressionSize = 0;
  return options;
}

// Chooses encodings the same way test::Encoder does, so encoded layouts stay
// comparable, while routing the chosen compressor into nested selection.
template <typename TInner>
class BenchEncodingSelectionPolicy
    : public nimble::EncodingSelectionPolicy<TInner> {
  using physicalType = typename nimble::TypeTraits<TInner>::physicalType;

 public:
  BenchEncodingSelectionPolicy(
      CompressionType compressionType,
      bool realNestedSelection)
      : compressionType_{compressionType},
        realNestedSelection_{realNestedSelection} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {
        .encodingType = nimble::EncodingType::Trivial,
        .compressionPolicyFactory = [this]() {
          return std::make_unique<BenchCompressPolicy>(compressionType_);
        }};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType parentEncodingType,
      nimble::NestedEncodingIdentifier nestedEncodingIdentifier,
      nimble::DataType type) override {
    // With realNestedSelection set, a sub-stream's encodings are chosen by the
    // writer's own cost-based factory rather than forced to Trivial, so
    // SubIntSplit exercises its per-section encoders. The one difference from
    // the writer is that the chosen compressor is passed down instead of
    // std::nullopt.
    //
    // The candidate list comes from nestedEncodingReadFactors, the same
    // function ManualEncodingSelectionPolicy::createImpl uses, with both
    // parentEncodingType and the nested identifier forwarded to it rather than
    // discarded: the augmented candidate list a SubIntSplit section gets is
    // keyed on the parent type, and forwarding it also removes the parent
    // encoding from the candidates, matching what the writer does.
    if (realNestedSelection_) {
      auto readFactors = nimble::nestedEncodingReadFactors(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          parentEncodingType,
          nestedEncodingIdentifier);
      // Withdraws the named encodings from what a section may be encoded as.
      // Empty, the default, leaves the writer's candidate list untouched.
      // Withdrawing from the split planner alone moves only where boundaries
      // fall; matching is by substring of toString(), so "Delta" also
      // withdraws DeltaBlock.
      if (!FLAGS_mlidc_sis_withdraw_nested_encodings.empty()) {
        const std::string& withdrawn =
            FLAGS_mlidc_sis_withdraw_nested_encodings;
        readFactors.erase(
            std::remove_if(
                readFactors.begin(),
                readFactors.end(),
                [&withdrawn](
                    const std::pair<nimble::EncodingType, float>& entry) {
                  return withdrawn.find(nimble::toString(entry.first)) !=
                      std::string::npos;
                }),
            readFactors.end());
      }
      // Holds sections to --mlidc_sis_allowed_encodings, which the split
      // planner is held to as well.
      if (const auto allowed = sisAllowedEncodings(); !allowed.empty()) {
        readFactors.erase(
            std::remove_if(
                readFactors.begin(),
                readFactors.end(),
                [&allowed](
                    const std::pair<nimble::EncodingType, float>& entry) {
                  return allowed.count(entry.first) == 0;
                }),
            readFactors.end());
      }
      // Built with its identifier, as ManualEncodingSelectionPolicy::createImpl
      // builds a writer's nested policy, so that the policy knows it selects
      // for a nested stream and honours subIntSplit.inNestedStreams.
      UNIQUE_PTR_FACTORY(
          type,
          nimble::ManualEncodingSelectionPolicy,
          std::move(readFactors),
          compressionOptionsFor(compressionType_),
          nestedEncodingIdentifier,
          std::nullopt);
    }
    UNIQUE_PTR_FACTORY(
        type,
        BenchEncodingSelectionPolicy,
        compressionType_,
        realNestedSelection_);
  }

 private:
  CompressionType compressionType_;
  bool realNestedSelection_;
};

// Encodes values with the given encoding, applying compressionType to the
// encoding's own stream and to any sub-streams it creates.
//
// `tuning` is SubIntSplit's configuration. It reaches E only where E takes
// one (see kEncodesWithTuning); every other encoding ignores it.
//
// `encodingConfig` is the encoding-specific configuration a selection policy
// would have attached. Empty for an ordinary encode; a caller measuring what
// one SubIntSplit split plan costs passes preserve-mode boundaries here, which
// is the only way to encode a *given* plan rather than one the encoder
// re-derives for itself. Routing that through this function instead of a
// second copy of it is deliberate: a copy drifts from the policy and
// compression wiring below, and a plan measured against a drifted copy is not
// being measured against the same encoder the drivers report.
template <typename E, typename T>
std::string_view encodeWithCompression(
    nimble::Buffer& buffer,
    const nimble::Vector<T>& values,
    CompressionType compressionType,
    const nimble::Encoding::Options& options,
    const subintsplit::TuningConfig& tuning,
    bool realNestedSelection,
    nimble::EncodingLayout::Config encodingConfig = {}) {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  auto physicalValues = std::span<const physicalType>(
      reinterpret_cast<const physicalType*>(values.data()), values.size());

  nimble::EncodingSelection<physicalType> selection{
      {.encodingType = test::EncodingTypeTraits<E>::encodingType,
       .encodingConfig = std::move(encodingConfig),
       .compressionPolicyFactory =
           [compressionType]() {
             return std::make_unique<BenchCompressPolicy>(compressionType);
           }},
      nimble::Statistics<physicalType>::create(physicalValues),
      std::make_unique<BenchEncodingSelectionPolicy<T>>(
          compressionType, realNestedSelection)};

  // Applied here, where every Nimble target's encode passes, rather than per
  // arm, so that one flag changes every arm alike.
  nimble::Encoding::Options flaggedOptions = options;
  subintsplit::TuningConfig flaggedTuning = tuning;
  if (FLAGS_mlidc_sis_section_threads != 0) {
    // One pool for the process, sized once from the flag.
    static folly::CPUThreadPoolExecutor sectionExecutor{
        FLAGS_mlidc_sis_section_threads};
    flaggedTuning.sectionExecutor = &sectionExecutor;
  }
  if (FLAGS_mlidc_selection_screen_rows != 0) {
    flaggedOptions.selectionScreenRows = FLAGS_mlidc_selection_screen_rows;
    flaggedOptions.selectionScreenMargin = FLAGS_mlidc_selection_screen_margin;
  }
  if constexpr (kEncodesWithTuning<E>) {
    return E::encode(
        selection, physicalValues, buffer, flaggedOptions, flaggedTuning);
  } else {
    return E::encode(selection, physicalValues, buffer, flaggedOptions);
  }
}

} // namespace facebook::nimble::mlidc

namespace facebook::nimble::test {

// Keys the encode cache only; the stream's actual encoding is whatever
// selection chose.
template <typename T>
struct EncodingTypeTraits<mlidc::AutoSelectedEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Trivial;
};

} // namespace facebook::nimble::test

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
