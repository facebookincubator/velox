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

#include "velox/dwio/nimble/common/Types.h"

#include <memory>
#include <string>

namespace facebook::nimble {

class Checksum {
 public:
  virtual ~Checksum() = default;
  virtual void update(std::string_view data) = 0;
  virtual uint64_t getChecksum(bool reset = false) = 0;
  virtual ChecksumType getType() const = 0;
};

class ChecksumFactory {
 public:
  static std::unique_ptr<Checksum> create(ChecksumType type);
};

} // namespace facebook::nimble
