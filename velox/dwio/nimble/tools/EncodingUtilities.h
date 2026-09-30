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

#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"

namespace facebook::nimble::tools {

enum class EncodingPropertyType {
  Compression,
  EncodedSize,
};

std::ostream& operator<<(std::ostream& out, EncodingPropertyType propertyType);

struct EncodingProperty {
  std::string value;
  std::string_view data{};
};

void traverseEncodings(
    std::string_view stream,
    std::function<bool(
        EncodingType,
        DataType,
        uint32_t /* level */,
        uint32_t /* index */,
        std::string /* nestedEncodingName */,
        std::unordered_map<
            EncodingPropertyType,
            EncodingProperty> /* properties */)> visitor);

std::string getStreamInputLabel(nimble::ChunkedStream& stream);
std::string getEncodingLabel(std::string_view stream);

/// One line per node of an encoding tree, for diffing what a change moved.
///
/// getEncodingLabel is written for a person and nests its children inside the
/// parent's line, which makes a node's position in the text depend on its
/// siblings. Two parsers of it have already misread a tree: one under-counted
/// switches by zipping node lists positionally, and one reported a column as
/// unchanged when a nested encoding under a Dictionary had in fact switched
/// and halved that column's decode throughput. This format is for scripts:
/// every node is one record, and every record carries the path that identifies
/// it, so a diff keys on identity rather than on order.
///
/// Tab-separated, with a leading `#` header naming the columns:
///
///   depth  path  index  encoding  dataType  bytes  compression
///
/// `path` is `/` at the root and slash-joined nested encoding names below it,
/// so a node is `/Section2/Indices/Baselines`. `bytes` is the node's whole
/// encoded span, its children included, so a parent's bytes minus its
/// children's is what that node spends on itself.
std::string getEncodingTreeLabel(std::string_view stream);

} // namespace facebook::nimble::tools
