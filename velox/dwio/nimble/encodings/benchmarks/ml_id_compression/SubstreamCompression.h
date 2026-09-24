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
#include <utility>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gflags/gflags.h>

DECLARE_string(mlidc_sis_withdraw_nested_encodings);
DECLARE_uint32(mlidc_selection_screen_rows);
DECLARE_double(mlidc_selection_screen_margin);
DECLARE_uint32(mlidc_sis_section_threads);

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
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
      // Built with its identifier, as ManualEncodingSelectionPolicy::createImpl
      // builds a writer's nested policy, so that the policy knows it selects
      // for a nested stream and honours subIntSplitInNestedStreams.
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
  if (FLAGS_mlidc_sis_section_threads != 0) {
    // One pool for the process, sized once from the flag.
    static folly::CPUThreadPoolExecutor sectionExecutor{
        FLAGS_mlidc_sis_section_threads};
    flaggedOptions.subIntSplitSectionExecutor = &sectionExecutor;
  }
  if (FLAGS_mlidc_selection_screen_rows != 0) {
    flaggedOptions.selectionScreenRows = FLAGS_mlidc_selection_screen_rows;
    flaggedOptions.selectionScreenMargin = FLAGS_mlidc_selection_screen_margin;
  }
  return E::encode(selection, physicalValues, buffer, flaggedOptions);
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
