/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/experimental/wave/dwio/StructColumnReader.h"

namespace facebook ::velox::wave {

bool StructColumnReader::isChildConstant(
    const velox::common::ScanSpec& childSpec) const {
  // Returns true if the child has a constant set in the ScanSpec, or if the
  // file doesn't have this child (in which case it will be treated as null).
  return childSpec.isConstant() ||
      // The below check is trying to determine if this is a missing field in a
      // struct that should be constant null.
      (!isRoot_ && // If we're in the root struct channel is meaningless in this
                   // context and it will be a null constant anyway if it's
                   // missing.
       childSpec.channel() !=
           velox::common::ScanSpec::kNoChannel && // This can happen if there's
                                                  // a filter on a subfield of a
                                                  // row type that doesn't exist
                                                  // in the output.
       fileType_->type()->kind() !=
           TypeKind::MAP && // If this is the case it means this is a flat map,
                            // so it can't have "missing" fields.
       childSpec.channel() >= fileType_->size());
}

// static
AbstractOperand* FOLLY_NULLABLE StructColumnReader::structOperand(
    const DefinesMap& defines,
    Path& path,
    bool isRoot,
    bool isNested) {
  if (isNested) {
    return nullptr;
  }
  auto* operand = pathToOperand(defines, path);
  if (isRoot || !operand) {
    return operand;
  }
  // The scan defines and uses each column once. More uses read the whole
  // struct.
  VELOX_CHECK_LE(
      operand->numUses, 1, "Reading a whole struct column is not supported");
  return nullptr;
}

void StructColumnReader::addChildren(
    velox::common::ScanSpec& scanSpec,
    Path& path,
    const DefinesMap& defines) {
  // A reader tree may be constructed while the ScanSpec is being used
  // for another read. This happens when the next stripe is being
  // prepared while the previous one is reading.
  const auto stableChildren = scanSpec.stableChildren();
  const auto& requestedRow = requestedType_->asRow();
  const auto& fileRow = fileType_->type()->asRow();
  for (const auto& childSpec : *stableChildren) {
    VELOX_CHECK(
        !childSpec->isConstant() || childSpec->constantValue()->isNullAt(0),
        "Non-null constant columns are not supported");
    const auto& name = childSpec->fieldName();
    const auto fileIdx = fileRow.getChildIdxIfExists(name);
    // A column that is only filtered on is not in the requested type.
    const auto requestedIdx = requestedRow.getChildIdxIfExists(name);
    TypePtr childRequestedType;
    if (requestedIdx.has_value()) {
      childRequestedType = requestedRow.childAt(requestedIdx.value());
    } else {
      VELOX_CHECK(
          fileIdx.has_value(),
          "Column {} is in neither the file nor the output",
          name);
      childRequestedType = fileRow.childAt(fileIdx.value());
    }
    const bool inFile = fileIdx.has_value() && !childSpec->isConstant();
    auto childFileType = fileIdx.has_value()
        ? fileType_->childAt(fileIdx.value())
        : dwio::common::TypeWithId::create(childRequestedType);
    path.push_back(std::make_unique<common::Subfield::NestedField>(name));
    addChild(makeChild(
        *childSpec,
        inFile ? static_cast<int32_t>(fileIdx.value()) : -1,
        childRequestedType,
        childFileType,
        path,
        defines));
    path.pop_back();
    childSpec->setSubscript(static_cast<int64_t>(children_.size()) - 1);
  }
}

void RepeatedColumnReader::addElements(
    FormatParams& params,
    velox::common::ScanSpec& scanSpec,
    Path& path,
    const DefinesMap& defines) {
  params.pushNested(formatData_.get());
  auto add = [&](const std::string& name, int32_t childIdx) {
    auto* childSpec = scanSpec.childByName(name);
    VELOX_CHECK_NOT_NULL(childSpec, "No ScanSpec for {}", name);
    addChild(makeChild(
        *childSpec,
        childIdx,
        requestedType_->childAt(childIdx),
        fileType_->childAt(childIdx),
        path,
        defines));
  };
  if (requestedType_->isArray()) {
    add(common::ScanSpec::kArrayElementsFieldName, 0);
  } else {
    add(common::ScanSpec::kMapKeysFieldName, 0);
    add(common::ScanSpec::kMapValuesFieldName, 1);
  }
  params.popNested();
}

} // namespace facebook::velox::wave
