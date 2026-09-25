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

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <folly/Conv.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

// What every node of an encoding tree costs, against what its selection was
// quoted for it.
//
// Nested children -- a Dictionary's indices, a FOR frame index, a Delta
// restatement stream -- are not like top-level sections: they are short,
// structurally regular, and often near-degenerate, so an estimator can score
// well on sections and be badly wrong on them. Such an error also hides by
// construction: on a small stream inside a multi-megabyte column, selection
// is nearly indifferent between candidates, so a bad estimate there costs
// almost nothing in bytes while a metadata stream that is read on every
// access can still cost a great deal in decode.
//
// The estimate is recomputed rather than captured from the writer, since
// select() does not know where in the tree it sits, so logging its result
// would need a positional join over the tree that has proven unreliable.
// Decoding each node and re-asking EncodingSizeEstimation is keyed by path
// by construction and needs no change to the writer.
namespace facebook::nimble::mlidc {

// Runs `fn` instantiated on the logical type behind `dataType`. Returns false
// for types this does not cover, which today means strings, since their
// estimators need a string buffer factory to decode against.
template <typename Fn>
bool dispatchNodeDataType(DataType dataType, Fn&& fn) {
  switch (dataType) {
    case DataType::Int8:
      fn.template operator()<int8_t>();
      return true;
    case DataType::Uint8:
      fn.template operator()<uint8_t>();
      return true;
    case DataType::Int16:
      fn.template operator()<int16_t>();
      return true;
    case DataType::Uint16:
      fn.template operator()<uint16_t>();
      return true;
    case DataType::Int32:
      fn.template operator()<int32_t>();
      return true;
    case DataType::Uint32:
      fn.template operator()<uint32_t>();
      return true;
    case DataType::Int64:
      fn.template operator()<int64_t>();
      return true;
    case DataType::Uint64:
      fn.template operator()<uint64_t>();
      return true;
    case DataType::Float:
      fn.template operator()<float>();
      return true;
    case DataType::Double:
      fn.template operator()<double>();
      return true;
    case DataType::Bool:
      fn.template operator()<bool>();
      return true;
    default:
      return false;
  }
}

// Whether a node carries the values of its parent or the bookkeeping that
// addresses them. Keyed on the nested encoding name traverseEncodings
// assigns, which is the encoding's own term for the child.
inline std::string_view encodingNodeKind(std::string_view nestedEncodingName) {
  if (nestedEncodingName.empty()) {
    return "root";
  }
  constexpr std::string_view kMetadata[] = {
      "Lengths",
      "Baselines",
      "BitWidths",
      "DataOffsets",
      "BitOffsets",
      "References",
      "Indices",
      "IsCommon",
      "IsRestatements",
      "ExceptionPositions",
      "PartitionOffsets",
      "PartitionSizes",
      "Nulls",
      "Sentinels"};
  for (const auto candidate : kMetadata) {
    if (candidate == nestedEncodingName) {
      return "metadata";
    }
  }
  return "payload";
}

/// One line per node: what it cost, what it was quoted, and the ratio.
///
/// Tab separated with a leading `#` header. `ratio` is actual over estimate
/// on the same convention the per-encoding figures use, so above one means
/// the node costs more than selection was told. An empty estimate means the
/// node's type is not covered by dispatchNodeDataType, or decoding it threw.
///
/// `pricedWith` and `exactBits` say which options priced each node: the
/// writer prices everything below a SubIntSplit under section options,
/// where FixedBitWidth packs at the exact bit width instead of rounding up
/// to a byte, and pricing a node under the wrong options silently inflates
/// its estimate. A run that states its own assumptions guards against that.
inline std::string describeEncodingNodeEstimates(
    std::string_view stream,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options) {
  std::string out =
      "#depth\tpath\tkind\tencoding\tdataType\trows\tactual\testimate"
      "\tratio\tpricedWith\texactBits\n";
  std::vector<std::string> path;
  // The encoding at each level of the current path, so a node can ask what
  // its ancestors are. Truncated and pushed alongside `path`, since the
  // traversal is depth-first and pre-order.
  std::vector<EncodingType> ancestry;

  // Options the writer would have priced this node under. SubIntSplit is
  // singled out because it is the one encoding that overrides its options
  // for everything beneath it: it derives section options once and hands
  // those to every section, so the whole subtree is encoded under them
  // rather than under the column's. The SubIntSplit node itself is priced
  // under column options -- it is the column's own encoding; only what is
  // below it is a section.
  const Encoding::Options sectionOptions =
      ::facebook::nimble::subintsplit::sectionEncodingOptions(options);
  const auto optionsForNode = [&](uint32_t level) -> const Encoding::Options& {
    for (uint32_t ancestor = 0; ancestor < level; ++ancestor) {
      if (ancestry[ancestor] == EncodingType::SubIntSplit ||
          ancestry[ancestor] == EncodingType::SubIntSplitReordered) {
        return sectionOptions;
      }
    }
    return options;
  };

  tools::traverseEncodings(
      stream,
      [&](EncodingType encodingType,
          DataType dataType,
          uint32_t level,
          uint32_t /*index*/,
          std::string nestedEncodingName,
          std::unordered_map<
              tools::EncodingPropertyType,
              tools::EncodingProperty> properties) -> bool {
        const auto kind = encodingNodeKind(nestedEncodingName);
        path.resize(level);
        path.push_back(std::move(nestedEncodingName));
        ancestry.resize(level);
        ancestry.push_back(encodingType);
        const auto& nodeOptions = optionsForNode(level);
        std::string joined;
        for (size_t i = 1; i < path.size(); ++i) {
          joined += "/";
          joined += path[i];
        }
        if (joined.empty()) {
          joined = "/";
        }

        const auto sizeProperty =
            properties.find(tools::EncodingPropertyType::EncodedSize);
        if (sizeProperty == properties.end() ||
            sizeProperty->second.data.empty()) {
          return true;
        }
        const std::string_view nodeStream = sizeProperty->second.data;
        const uint64_t actual = nodeStream.size();

        std::optional<uint64_t> estimate;
        uint32_t rows = 0;
        dispatchNodeDataType(dataType, [&]<typename T>() {
          using P = typename TypeTraits<T>::physicalType;
          try {
            auto encoding = EncodingFactory().create(
                pool,
                nodeStream,
                [](uint32_t) -> void* { return nullptr; },
                nodeOptions);
            if (encoding == nullptr) {
              return;
            }
            rows = encoding->rowCount();
            Vector<P> values{&pool, rows};
            encoding->materialize(rows, values.data());
            const std::span<const P> span{values.data(), rows};
            const auto statistics = Statistics<P>::create(span);
            // The same function selection consults, asked for the encoding
            // this node actually got, over the values it actually holds.
            // Fully qualified on purpose: several headers in this namespace
            // declare an `mlidc::detail`, so an unqualified `detail::` here
            // would resolve to that one instead of nimble's.
            estimate = ::facebook::nimble::detail::EncodingSizeEstimation<
                T>::estimateSize(encodingType, span, statistics, nodeOptions);
          } catch (...) {
            // A node whose type cannot be decoded here is reported without an
            // estimate rather than skipped, so its bytes still appear.
          }
        });

        out += folly::to<std::string>(
            level,
            "\t",
            joined,
            "\t",
            kind,
            "\t",
            toString(encodingType),
            "\t",
            toString(dataType),
            "\t",
            rows,
            "\t",
            actual,
            "\t");
        if (estimate.has_value()) {
          out += folly::to<std::string>(estimate.value(), "\t");
          out += estimate.value() > 0
              ? folly::to<std::string>(
                    static_cast<double>(actual) /
                    static_cast<double>(estimate.value()))
              : std::string{};
        } else {
          out += "\t";
        }
        // Which options priced this node, stated rather than assumed.
        out += folly::to<std::string>(
            "\t",
            &nodeOptions == &sectionOptions ? "section" : "column",
            "\t",
            nodeOptions.fixedBitWidthUseExactBits ? 1 : 0,
            "\n");
        return true;
      });

  return out;
}

namespace detail {

// One priced alternative for a stream: an encoding and what it was quoted.
struct PricedEncoding {
  EncodingType encoding;
  double bytes;
  // What selection compares: bytes times the effective read factor. Equal
  // to bytes for the planner's models, which weigh no factor.
  double cost;
};

// Selection's quotes for `values` under `sectionOptions`, cheapest cost first,
// over the candidates a SubIntSplit section is offered.
template <typename T>
std::vector<PricedEncoding> sectionSelectionQuotes(
    std::span<const T> values,
    const Encoding::Options& sectionOptions) {
  const auto statistics = Statistics<T>::create(values);
  const auto fixedBitWidthBytes =
      ::facebook::nimble::detail::EncodingSizeEstimation<T>::estimateSize(
          EncodingType::FixedBitWidth, values, statistics, sectionOptions);
  std::vector<PricedEncoding> quotes;
  for (const auto& [encoding, readFactor] : nestedEncodingReadFactors(
           ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
           EncodingType::SubIntSplit)) {
    const auto estimate =
        ::facebook::nimble::detail::EncodingSizeEstimation<T>::estimateSize(
            encoding, values, statistics, sectionOptions);
    if (!estimate.has_value()) {
      continue;
    }
    const float factor = effectiveReadFactor(
        encoding, readFactor, estimate.value(), fixedBitWidthBytes);
    quotes.push_back(
        {encoding,
         static_cast<double>(estimate.value()),
         static_cast<double>(estimate.value() * factor)});
  }
  std::stable_sort(
      quotes.begin(), quotes.end(), [](const auto& a, const auto& b) {
        return a.cost < b.cost;
      });
  return quotes;
}

// The split planner's model quotes for `values`, a `bitWidth`-bit range,
// cheapest first, from the sample the writer would draw and scaled to the
// stream, under the production planner's inventory.
inline std::vector<PricedEncoding> plannerModelQuotes(
    std::span<const uint64_t> values,
    int bitWidth) {
  namespace sis = ::facebook::nimble::subintsplit;
  std::vector<uint64_t> sample;
  sis::sampleIntoU64<uint64_t>(values, sample, sis::defaultSamplerConfig());
  if (sample.empty()) {
    return {};
  }
  sis::MetricCollector collector;
  const auto metrics =
      collector.compute(sample, sis::allCostModelRequiredFlags());
  const double scale =
      static_cast<double>(values.size()) / static_cast<double>(sample.size());
  std::vector<PricedEncoding> quotes;
  for (const auto encoding :
       {EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::Constant,
        EncodingType::MainlyConstant,
        EncodingType::RLE,
        EncodingType::Varint,
        EncodingType::Dictionary,
        EncodingType::SimdForBitpack,
        EncodingType::PFOR,
        EncodingType::BlockBitPacking,
        EncodingType::Delta,
        EncodingType::FOR,
        EncodingType::FrequencyPartition}) {
    const auto cost = sis::bestSectionCost(
        metrics,
        sample.size(),
        values.size(),
        bitWidth,
        sample,
        sis::AllowedEncodings{encoding},
        /*allowHuffman=*/false,
        /*allowDeltaBlock=*/false,
        sis::DecodeCostWeighting{});
    if (std::isfinite(cost.sizeBits)) {
      const double bytes = cost.sizeBits * scale / 8.0;
      quotes.push_back({encoding, bytes, bytes});
    }
  }
  std::stable_sort(
      quotes.begin(), quotes.end(), [](const auto& a, const auto& b) {
        return a.cost < b.cost;
      });
  return quotes;
}

inline std::string formatQuote(
    const std::vector<PricedEncoding>& quotes,
    size_t rank) {
  if (rank >= quotes.size()) {
    return "\t";
  }
  return folly::to<std::string>(
      toString(quotes[rank].encoding),
      "\t",
      static_cast<uint64_t>(quotes[rank].bytes));
}

inline std::string formatQuotes(const std::vector<PricedEncoding>& quotes) {
  std::string out;
  for (const auto& quote : quotes) {
    out += folly::to<std::string>(
        out.empty() ? "" : ";",
        toString(quote.encoding),
        "=",
        static_cast<uint64_t>(quote.bytes),
        "/",
        static_cast<uint64_t>(quote.cost));
  }
  return out;
}

// Decodes every value of one SubIntSplit section, widened to 64 bits.
inline std::vector<uint64_t> decodeSectionValues(
    const ::facebook::nimble::subintsplit::StoredSection& section,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& sectionOptions) {
  auto encoding = EncodingFactory().create(
      pool,
      section.stream,
      [](uint32_t) -> void* { return nullptr; },
      sectionOptions);
  const uint32_t rows = encoding->rowCount();
  std::vector<uint64_t> values(rows);
  const auto widen = [&]<typename S>() {
    std::vector<S> narrow(rows);
    encoding->materialize(rows, narrow.data());
    std::copy(narrow.begin(), narrow.end(), values.begin());
  };
  switch (section.storageBytes) {
    case 1:
      widen.template operator()<uint8_t>();
      break;
    case 2:
      widen.template operator()<uint16_t>();
      break;
    case 4:
      widen.template operator()<uint32_t>();
      break;
    default:
      widen.template operator()<uint64_t>();
      break;
  }
  return values;
}

// Selection's quotes for section values held at `storageBytes`.
inline std::vector<PricedEncoding> sectionSelectionQuotesAt(
    const std::vector<uint64_t>& values,
    uint8_t storageBytes,
    const Encoding::Options& sectionOptions) {
  const auto narrowed = [&]<typename S>() {
    std::vector<S> narrow(values.begin(), values.end());
    return sectionSelectionQuotes<S>(
        std::span<const S>(narrow), sectionOptions);
  };
  switch (storageBytes) {
    case 1:
      return narrowed.template operator()<uint8_t>();
    case 2:
      return narrowed.template operator()<uint16_t>();
    case 4:
      return narrowed.template operator()<uint32_t>();
    default:
      return narrowed.template operator()<uint64_t>();
  }
}

} // namespace detail

/// Why a SubIntSplit stream's plan looks the way it does, one line per
/// section and one for the whole value, empty when `stream` is not
/// SubIntSplit.
///
/// For each section: its bits, transform and encoding, the bytes it took,
/// the two cheapest candidates as section selection costs them, and the two
/// cheapest of the split planner's models for the same bits. `all` lists
/// every selection quote as encoding=estimate/cost. The `whole` line prices
/// the value as one section, the alternative every split was chosen over;
/// it is omitted when a transform reorders rows, since the stored sections
/// no longer reassemble the values.
inline std::string describeSubIntSplitSectionChoices(
    std::string_view stream,
    velox::memory::MemoryPool& pool) {
  namespace sis = ::facebook::nimble::subintsplit;
  auto root = EncodingFactory().create(
      pool,
      stream,
      [](uint32_t) -> void* { return nullptr; },
      Encoding::Options{});
  if (root == nullptr ||
      (root->encodingType() != EncodingType::SubIntSplit &&
       root->encodingType() != EncodingType::SubIntSplitReordered)) {
    return {};
  }
  const uint32_t rows = root->rowCount();
  const int valueBits = root->dataType() == DataType::Int32 ||
          root->dataType() == DataType::Uint32 ||
          root->dataType() == DataType::Float
      ? 32
      : 64;
  sis::TransformInfo transformInfo;
  sis::RowFrame rowFrame;
  const auto sections =
      sis::parseSections(stream, root->dataOffset(), &transformInfo, &rowFrame);
  const Encoding::Options sectionOptions =
      sis::sectionEncodingOptions(Encoding::Options{});

  // Which fit produced the frame, refitted on the decoded column since the
  // stream records only the line: "line" when fitRowFrame reproduces it,
  // "step" when fitStepFrame does, "none" without a frame.
  std::string frameKind = "none";
  if (rowFrame.active()) {
    const auto kindOf = [&]<typename P>() {
      std::vector<P> original(rows);
      root->materialize(rows, original.data());
      const std::span<const P> span{original};
      const auto line = sis::fitRowFrame(span);
      if (line.active() && line.slope == rowFrame.slope &&
          line.base == rowFrame.base) {
        return std::string("line");
      }
      const auto step = sis::fitStepFrame(span);
      return step.slope == rowFrame.slope && step.base == rowFrame.base
          ? std::string("step")
          : std::string("unknown");
    };
    frameKind = valueBits == 32 ? kindOf.template operator()<uint32_t>()
                                : kindOf.template operator()<uint64_t>();
  }
  std::string out = folly::to<std::string>(
      "#frameSlope=",
      rowFrame.slope,
      " frameBase=",
      rowFrame.base,
      " frameKind=",
      frameKind,
      " keySection=",
      transformInfo.anyTransform() ? static_cast<int>(transformInfo.keySection)
                                   : -1,
      "\n#section\tbitStart\tbitEnd\ttransform\tencoding\tactual"
      "\tselPick\tselPickEstimate\tselRunnerUp\tselRunnerUpEstimate"
      "\tplannerPick\tplannerPickBytes\tplannerRunnerUp\tplannerRunnerUpBytes"
      "\tall\n");
  std::vector<uint64_t> wholeValues(rows, 0);
  for (size_t s = 0; s < sections.size(); ++s) {
    const auto& section = sections[s];
    const auto values =
        detail::decodeSectionValues(section, pool, sectionOptions);
    const uint8_t transformId = transformInfo.anyTransform()
        ? transformInfo.transformIds[s]
        : uint8_t{0};
    for (uint32_t row = 0; row < rows && row < values.size(); ++row) {
      wholeValues[row] |= values[row] << section.bitStart;
    }
    const auto selection = detail::sectionSelectionQuotesAt(
        values, section.storageBytes, sectionOptions);
    const auto planner = detail::plannerModelQuotes(
        values, section.bitEnd - section.bitStart + 1);
    auto sectionEncoding = EncodingFactory().create(
        pool,
        section.stream,
        [](uint32_t) -> void* { return nullptr; },
        sectionOptions);
    out += folly::to<std::string>(
        s,
        "\t",
        section.bitStart,
        "\t",
        section.bitEnd,
        "\t",
        static_cast<int>(transformId),
        "\t",
        toString(sectionEncoding->encodingType()),
        "\t",
        section.stream.size(),
        "\t",
        detail::formatQuote(selection, 0),
        "\t",
        detail::formatQuote(selection, 1),
        "\t",
        detail::formatQuote(planner, 0),
        "\t",
        detail::formatQuote(planner, 1),
        "\t",
        detail::formatQuotes(selection),
        "\n");
  }
  if (!transformInfo.anyTransform()) {
    const auto selection = detail::sectionSelectionQuotesAt(
        wholeValues, static_cast<uint8_t>(valueBits / 8), sectionOptions);
    const auto planner = detail::plannerModelQuotes(wholeValues, valueBits);
    out += folly::to<std::string>(
        "whole\t0\t",
        valueBits - 1,
        "\t0\t\t\t",
        detail::formatQuote(selection, 0),
        "\t",
        detail::formatQuote(selection, 1),
        "\t",
        detail::formatQuote(planner, 0),
        "\t",
        detail::formatQuote(planner, 1),
        "\t",
        detail::formatQuotes(selection),
        "\n");
  }
  return out;
}

} // namespace facebook::nimble::mlidc
