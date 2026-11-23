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
#include <ostream>

#include "velox/common/file/File.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::tools {

class NimbleDumpLib {
 public:
  NimbleDumpLib(
      const std::string& filePath,
      bool enableColors,
      std::ostream& ostream);

  NimbleDumpLib(
      std::shared_ptr<velox::ReadFile> file,
      bool enableColors,
      std::ostream& ostream);

  void emitInfo();
  void emitSchema(bool collapseFlatMap = true);
  void emitStripes(bool noHeader);
  void emitStreams(
      bool noHeader,
      bool flatmapKeys,
      bool rawSize,
      bool showInMapStream,
      std::optional<uint32_t> stripeId);
  void
  emitHistogram(bool topLevel, bool noHeader, std::optional<uint32_t> stripeId);
  void emitContent(
      uint32_t streamId,
      std::optional<uint32_t> stripeId,
      const std::string& separator);
  void emitBinary(
      std::function<std::unique_ptr<std::ostream>()> outputFactory,
      uint32_t streamId,
      uint32_t stripeId);
  void emitLayout(bool noHeader, bool compressed);
  void emitFileLayout(bool noHeader);
  void emitStripesMetadata(bool noHeader);
  void emitStripeGroupsMetadata(bool noHeader);
  void emitOptionalSectionsMetadata(bool noHeader);
  void emitIndex();
  void emitStats(bool noHeader);

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::ReadFile> file_;
  std::ostream& ostream_;
  bool enableColors_;
};
} // namespace facebook::nimble::tools
