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

#include <unordered_map>
#include "folly/CppAttributes.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"

namespace facebook::nimble {

class EncodingLayoutTree {
 public:
  using StreamIdentifier = uint8_t;

  struct StreamIdentifiers {
    struct Scalar {
      constexpr static StreamIdentifier ScalarStream = 0;
    };
    struct TimestampMicroNano {
      constexpr static StreamIdentifier MicrosStream = 0;
      constexpr static StreamIdentifier NanosStream = 1;
    };
    struct Array {
      constexpr static StreamIdentifier LengthsStream = 0;
    };
    struct Map {
      constexpr static StreamIdentifier LengthsStream = 0;
    };
    struct Row {
      constexpr static StreamIdentifier NullsStream = 0;
    };
    struct FlatMap {
      constexpr static StreamIdentifier NullsStream = 0;
    };
    struct ArrayWithOffsets {
      constexpr static StreamIdentifier OffsetsStream = 0;
      constexpr static StreamIdentifier LengthsStream = 1;
    };
    struct SlidingWindowMap {
      constexpr static StreamIdentifier OffsetsStream = 0;
      constexpr static StreamIdentifier LengthsStream = 1;
    };
  };

  EncodingLayoutTree(
      Kind schemaKind,
      std::unordered_map<StreamIdentifier, EncodingLayout> encodingLayouts,
      std::string name,
      std::vector<EncodingLayoutTree> children = {});

  Kind schemaKind() const;
  const EncodingLayout* encodingLayout(StreamIdentifier) const;
  const std::string& name() const;
  uint32_t childrenCount() const;
  const EncodingLayoutTree& child(uint32_t index) const;

  uint32_t serialize(std::span<char> output) const;
  static EncodingLayoutTree create(std::string_view tree);

  std::vector<StreamIdentifier> encodingLayoutIdentifiers() const;

 private:
  const Kind schemaKind_;
  const std::unordered_map<StreamIdentifier, EncodingLayout> encodingLayouts_;
  const std::string name_;
  const std::vector<EncodingLayoutTree> children_;
};

} // namespace facebook::nimble
