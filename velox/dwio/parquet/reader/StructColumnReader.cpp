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

#include "velox/dwio/parquet/reader/StructColumnReader.h"

namespace facebook::velox::parquet {

void StructColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  for (auto& child : children_) {
    if (auto structChild = dynamic_cast<StructColumnReader*>(child.get())) {
      structChild->enqueueRowGroup(index, input);
    } else {
      child->formatData().as<ParquetData>().enqueueRowGroup(index, input);
    }
  }
}

void StructColumnReader::seekToRowGroup(uint32_t index) {
  readOffset_ = 0;
  for (auto& child : children_) {
    child->seekToRowGroup(index);
  }
}

bool StructColumnReader::filterMatches(const RowGroup& rowGroup) {
  return true;
#if 0
  bool matched = true;

  auto& childSpecs = scanSpec_->children();
  assert(!children_.empty());
  for (size_t i = 0; i < childSpecs.size(); ++i) {
    auto& childSpec = childSpecs[i];
    if (childSpec->isConstant()) {
      // TODO: match constant
      continue;
    }
    auto fieldIndex = childSpec->subscript();
    auto reader = children_.at(fieldIndex).get();
    //    auto colName = childSpec->fieldName();

    if (!reader->filterMatches(rowGroup)) {
      matched = false;
      break;
    }
  }
  return matched;
#endif
}

} // namespace facebook::velox::parquet
