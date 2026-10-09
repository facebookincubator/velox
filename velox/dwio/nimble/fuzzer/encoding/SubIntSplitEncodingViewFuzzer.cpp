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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "security/lionhead/utils/lib_ftest/ftest.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {
namespace {

velox::memory::MemoryPool& memoryPool() {
  static const auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  return *pool;
}

template <typename Fuzzer>
EncodingType childEncoding(Fuzzer& fuzzer) {
  switch (fuzzer.u8_range("child_encoding", 0, 2)) {
    case 0:
      return EncodingType::Constant;
    case 1:
      return EncodingType::Trivial;
    case 2:
      return EncodingType::FixedBitWidth;
  }
  NIMBLE_UNREACHABLE("Invalid generated child encoding.");
}

template <typename T, typename Fuzzer>
void fuzzView(Fuzzer& fuzzer) {
  constexpr uint8_t kNumBits{sizeof(T) * 8};
  const auto numRows = fuzzer.u16_range("num_rows", 1, 256);
  const auto numSections = fuzzer.u8_range("num_sections", 1, 4);

  std::vector<subintsplit::SectionPlan> sections;
  std::vector<EncodingType> childEncodings;
  sections.reserve(numSections);
  childEncodings.reserve(numSections);
  uint8_t bitStart{0};
  for (uint8_t sectionIndex{0}; sectionIndex < numSections; ++sectionIndex) {
    const auto sectionsRemaining = numSections - sectionIndex;
    const auto bitsRemaining = kNumBits - bitStart;
    const auto bitWidth = sectionsRemaining == 1
        ? bitsRemaining
        : fuzzer.u8_range(
              "section_width", 1, bitsRemaining - sectionsRemaining + 1);
    sections.push_back(
        {.bitStart = bitStart,
         .bitEnd = static_cast<uint8_t>(bitStart + bitWidth - 1)});
    childEncodings.push_back(childEncoding(fuzzer));
    bitStart += bitWidth;
  }

  std::vector<T> constantSectionValues;
  constantSectionValues.reserve(numSections);
  for (size_t sectionIndex{0}; sectionIndex < numSections; ++sectionIndex) {
    constantSectionValues.push_back(static_cast<T>(fuzzer.u64("constant")));
  }

  std::vector<T> values(numRows, 0);
  for (size_t sectionIndex{0}; sectionIndex < numSections; ++sectionIndex) {
    const auto& section = sections[sectionIndex];
    const auto bitWidth = section.bitEnd - section.bitStart + 1;
    const T mask = bitWidth == kNumBits
        ? std::numeric_limits<T>::max()
        : static_cast<T>((uint64_t{1} << bitWidth) - 1);
    for (auto& value : values) {
      const T sectionValue =
          childEncodings[sectionIndex] == EncodingType::Constant
          ? constantSectionValues[sectionIndex]
          : static_cast<T>(fuzzer.u64("section_value"));
      value |= static_cast<T>((sectionValue & mask) << section.bitStart);
    }
  }

  std::vector<std::optional<const EncodingLayout>> children;
  children.reserve(numSections);
  for (const auto encoding : childEncodings) {
    children.emplace_back(
        EncodingLayout{encoding, {}, CompressionType::Uncompressed});
  }
  EncodingLayout layout{
      EncodingType::SubIntSplit,
      EncodingLayout::Config{subintsplit::makePreserveSplitConfig(sections)},
      CompressionType::Uncompressed,
      std::move(children),
  };
  const EncodingSelectionPolicyCreator leafPolicyCreator =
      [](DataType dataType) {
        return ManualEncodingSelectionPolicyFactory{
            ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
            std::nullopt}
            .createPolicy(dataType);
      };
  Buffer buffer{memoryPool()};
  const auto encoded = EncodingFactory::encode<T>(
      std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          std::move(layout), std::nullopt, leafPolicyCreator),
      values,
      buffer);
  const auto view = createEncodingView(encoded, &memoryPool());

  for (uint32_t row{0}; row < numRows; ++row) {
    T actual{0};
    view->readAt(row, &actual);
    CHECK_EQ(actual, values[row]);
  }

  std::vector<T> contiguous(numRows);
  view->read(0, numRows, contiguous.data());
  CHECK(contiguous == values);

  const auto numIndices = fuzzer.u16_range("num_indices", 0, 128);
  std::vector<uint32_t> indices;
  indices.reserve(numIndices);
  for (uint16_t i{0}; i < numIndices; ++i) {
    indices.push_back(fuzzer.u32_range("index", 0, numRows - 1));
  }
  if (fuzzer.boolean("ordered")) {
    std::sort(indices.begin(), indices.end());
  }
  std::vector<T> indexed(numIndices);
  view->readAt(indices, indexed.data());
  for (size_t i{0}; i < indices.size(); ++i) {
    CHECK_EQ(indexed[i], values[indices[i]]);
  }
}

} // namespace
} // namespace facebook::nimble

FUZZ(SubIntSplitEncodingView, ReconstructsGeneratedLayouts) {
  if (f.boolean("use_64_bits")) {
    facebook::nimble::fuzzView<uint64_t>(f);
  } else {
    facebook::nimble::fuzzView<uint32_t>(f);
  }
}
