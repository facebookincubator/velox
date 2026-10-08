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

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"

namespace facebook::nimble {

class EncodingLayout {
 public:
  /// Used to pass encoding-specific configs.
  struct Config {
    Config() = default;
    explicit Config(std::unordered_map<std::string, std::string> configs)
        : configs_{std::move(configs)} {}

    /// Returns a config value for the given key, or std::nullopt if not found.
    std::optional<std::string> get(const std::string& key) const;

    const std::unordered_map<std::string, std::string>& values() const {
      return configs_;
    }

   private:
    std::unordered_map<std::string, std::string> configs_;
  };

  EncodingLayout(
      EncodingType encodingType,
      Config encodingConfig,
      CompressionType compressionType,
      std::vector<std::optional<const EncodingLayout>> children = {});

  EncodingType encodingType() const;

  CompressionType compressionType() const;

  uint8_t childrenCount() const;

  const std::optional<const EncodingLayout>& child(
      NestedEncodingIdentifier identifier) const;

  // Returns the encoding-specific configuration.
  const Config& config() const;

  int32_t serialize(std::span<char> output) const;

  static std::pair<EncodingLayout, uint32_t> create(std::string_view encoding);

 private:
  EncodingType encodingType_;

  Config encodingConfig_;

  CompressionType compressionType_;

  std::vector<std::optional<const EncodingLayout>> children_;
};

class EncodingLayoutCapture {
 public:
  /// Captures an encoding tree from an encoded stream.
  /// It traverses the encoding headers in the stream and produces a serialized
  /// encoding tree layout.
  /// |encoding| - The serialized encoding
  static EncodingLayout capture(
      std::string_view encoding,
      const Encoding::Options& options);
};

} // namespace facebook::nimble
