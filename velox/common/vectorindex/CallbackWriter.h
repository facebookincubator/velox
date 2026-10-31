/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <string_view>

#include <faiss/impl/io.h>
#include <folly/Function.h>

namespace facebook::velox::common::vectorindex {

/// Passes each chunk faiss::write_index writes to a callback, so the caller
/// can place the bytes without an intermediate copy.
class CallbackWriter : public faiss::IOWriter {
 public:
  explicit CallbackWriter(folly::FunctionRef<void(std::string_view)> write)
      : write_{write} {}

  size_t operator()(const void* ptr, size_t size, size_t numItems) override {
    write_(std::string_view(static_cast<const char*>(ptr), size * numItems));
    return numItems;
  }

 private:
  folly::FunctionRef<void(std::string_view)> write_;
};

} // namespace facebook::velox::common::vectorindex
