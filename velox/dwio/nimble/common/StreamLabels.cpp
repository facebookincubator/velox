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
#include "velox/dwio/nimble/common/StreamLabels.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

namespace {
void addLabels(
    const std::shared_ptr<const Type>& node,
    size_t labelIndex,
    const std::string& name,
    std::vector<std::string>& labels,
    std::vector<size_t>& offsetToLabel) {
  switch (node->kind()) {
    case Kind::Scalar: {
      const auto& scalar = node->asScalar();
      const auto offset = scalar.scalarDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(offsetToLabel.size(), offset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      offsetToLabel[offset] = labels.size() - 1;
      break;
    }
    case Kind::TimestampMicroNano: {
      const auto& timestamp = node->asTimestampMicroNano();
      const auto microsOffset = timestamp.microsDescriptor().offset();
      const auto nanosOffset = timestamp.nanosDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), microsOffset, "Unexpected micros offset.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), nanosOffset, "Unexpected nanos offset.");
      // linter does not recognize NIMBLE_DCHECK_LT as sufficient checks
      // @lint-ignore CLANGTIDY facebook-hte-ParameterUncheckedArrayBounds
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[microsOffset] = labelIndex;
      offsetToLabel[nanosOffset] = labelIndex;
      break;
    }
    case Kind::Array: {
      const auto& array = node->asArray();
      const auto offset = array.lengthsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(offsetToLabel.size(), offset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[offset] = labelIndex;
      addLabels(array.elements(), labelIndex, "", labels, offsetToLabel);
      break;
    }
    case Kind::Map: {
      const auto& map = node->asMap();
      const auto offset = map.lengthsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(offsetToLabel.size(), offset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[offset] = labelIndex;
      addLabels(map.keys(), labelIndex, "", labels, offsetToLabel);
      addLabels(map.values(), labelIndex, "", labels, offsetToLabel);
      break;
    }
    case Kind::SlidingWindowMap: {
      const auto& map = node->asSlidingWindowMap();
      const auto offsetsOffset = map.offsetsDescriptor().offset();
      const auto lengthsOffset = map.lengthsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), offsetsOffset, "Unexpected offset.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), lengthsOffset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[offsetsOffset] = labelIndex;
      offsetToLabel[lengthsOffset] = labelIndex;
      addLabels(map.keys(), labelIndex, "", labels, offsetToLabel);
      addLabels(map.values(), labelIndex, "", labels, offsetToLabel);
      break;
    }
    case Kind::Row: {
      const auto& row = node->asRow();
      const auto offset = row.nullsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(offsetToLabel.size(), offset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name + "/");
      labelIndex = labels.size() - 1;
      offsetToLabel[offset] = labelIndex;
      for (auto i = 0; i < row.childrenCount(); ++i) {
        addLabels(
            row.childAt(i),
            labelIndex,
            folly::to<std::string>(i),
            labels,
            offsetToLabel);
      }
      break;
    }
    case Kind::FlatMap: {
      const auto& map = node->asFlatMap();
      const auto offset = map.nullsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(offsetToLabel.size(), offset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[offset] = labelIndex;
      for (auto i = 0; i < map.childrenCount(); ++i) {
        const auto inMapOffset = map.inMapDescriptorAt(i).offset();
        NIMBLE_DCHECK_GT(
            offsetToLabel.size(), inMapOffset, "Unexpected offset.");
        labels.emplace_back(labels[labelIndex] + "/" + map.nameAt(i));
        offsetToLabel[inMapOffset] = labels.size() - 1;
      }
      for (auto i = 0; i < map.childrenCount(); ++i) {
        addLabels(
            map.childAt(i),
            offsetToLabel[map.inMapDescriptorAt(i).offset()],
            "",
            labels,
            offsetToLabel);
      }
      break;
    }
    case Kind::ArrayWithOffsets: {
      const auto& array = node->asArrayWithOffsets();
      const auto offsetsOffset = array.offsetsDescriptor().offset();
      const auto lengthsOffset = array.lengthsDescriptor().offset();
      NIMBLE_DCHECK_LT(labelIndex, labels.size(), "Unexpected label index.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), offsetsOffset, "Unexpected offset.");
      NIMBLE_DCHECK_GT(
          offsetToLabel.size(), lengthsOffset, "Unexpected offset.");
      labels.emplace_back(labels[labelIndex] + name);
      labelIndex = labels.size() - 1;
      offsetToLabel[offsetsOffset] = labelIndex;
      offsetToLabel[lengthsOffset] = labelIndex;
      addLabels(array.elements(), labelIndex, "", labels, offsetToLabel);
      break;
    }
  }
}
} // namespace

StreamLabels::StreamLabels(const std::shared_ptr<const Type>& root) {
  size_t maxOffset = 0;
  size_t labelCount = 1;
  SchemaReader::traverseSchema(
      root,
      [&](uint32_t level,
          const Type& type,
          const SchemaReader::NodeInfo& info) {
        ++labelCount;
        switch (type.kind()) {
          case Kind::Scalar: {
            maxOffset = std::max<size_t>(
                maxOffset, type.asScalar().scalarDescriptor().offset());
            break;
          }
          case Kind::TimestampMicroNano: {
            maxOffset = std::max<size_t>(
                maxOffset,
                type.asTimestampMicroNano().microsDescriptor().offset());
            maxOffset = std::max<size_t>(
                maxOffset,
                type.asTimestampMicroNano().nanosDescriptor().offset());
            break;
          }
          case Kind::Array: {
            maxOffset = std::max<size_t>(
                maxOffset, type.asArray().lengthsDescriptor().offset());
            break;
          }
          case Kind::Map: {
            maxOffset = std::max<size_t>(
                maxOffset, type.asMap().lengthsDescriptor().offset());
            break;
          }
          case Kind::SlidingWindowMap: {
            const auto& map = type.asSlidingWindowMap();
            maxOffset =
                std::max<size_t>(maxOffset, map.offsetsDescriptor().offset());
            maxOffset =
                std::max<size_t>(maxOffset, map.lengthsDescriptor().offset());
            break;
          }
          case Kind::Row: {
            maxOffset = std::max<size_t>(
                maxOffset, type.asRow().nullsDescriptor().offset());
            break;
          }
          case Kind::FlatMap: {
            const auto& map = type.asFlatMap();
            maxOffset =
                std::max<size_t>(maxOffset, map.nullsDescriptor().offset());
            for (auto i = 0; i < map.childrenCount(); ++i) {
              maxOffset = std::max<size_t>(
                  maxOffset, map.inMapDescriptorAt(i).offset());
            }
            labelCount += map.childrenCount();
            break;
          }
          case Kind::ArrayWithOffsets: {
            const auto& array = type.asArrayWithOffsets();
            maxOffset =
                std::max<size_t>(maxOffset, array.offsetsDescriptor().offset());
            maxOffset =
                std::max<size_t>(maxOffset, array.lengthsDescriptor().offset());
            break;
          }
        }
      });

  // Populate labels
  labels_.reserve(labelCount + 1);
  offsetToLabel_.resize(maxOffset + 1);

  labels_.emplace_back("");
  addLabels(root, 0, "", labels_, offsetToLabel_);
}

std::string_view StreamLabels::streamLabel(offset_size offset) const {
  NIMBLE_CHECK_LT(offset, offsetToLabel_.size(), "Stream offset out of range");
  return labels_[offsetToLabel_[offset]];
}

} // namespace facebook::nimble
