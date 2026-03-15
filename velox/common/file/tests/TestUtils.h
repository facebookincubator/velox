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

#pragma once

#include "velox/common/file/File.h"

namespace facebook::velox::tests::utils {

std::vector<std::string> iobufsToStrings(
    const std::vector<folly::IOBuf>& iobufs);

/// Wraps InMemoryReadFile with pread call counting for test assertions.
class CountingReadFile : public InMemoryReadFile {
 public:
  using InMemoryReadFile::InMemoryReadFile;

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override {
    ++numReads_;
    return InMemoryReadFile::pread(offset, length, buf, context);
  }

  uint64_t numReads() const {
    return numReads_;
  }

 private:
  mutable std::atomic_uint64_t numReads_{0};
};

} // namespace facebook::velox::tests::utils
