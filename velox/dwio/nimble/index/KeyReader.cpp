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
#include "velox/dwio/nimble/index/KeyReader.h"

#include <utility>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/KeyEncoding.h"

namespace facebook::nimble::index {

std::unique_ptr<KeyReader> createFlatKeyReader(
    std::string_view encodedKeys,
    std::function<void*(uint32_t)> stringBufferFactory,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  return std::make_unique<FlatKeyReader>(
      KeyEncoding::create(*pool, encodedKeys, std::move(stringBufferFactory)));
}

FlatKeyReader::FlatKeyReader(std::unique_ptr<KeyEncoding> encoding)
    : encoding_{std::move(encoding)} {
  NIMBLE_CHECK_NOT_NULL(encoding_);
}

FlatKeyReader::~FlatKeyReader() = default;

std::optional<uint32_t> FlatKeyReader::seek(
    std::string_view value,
    bool inclusive) const {
  return encoding_->seek(value, inclusive);
}

std::string FlatKeyReader::get(uint32_t row) const {
  return encoding_->get(row);
}

std::vector<std::string> FlatKeyReader::materialize(
    uint32_t startRow,
    uint32_t count) const {
  return encoding_->materialize(startRow, count);
}

std::unique_ptr<KeyCursor> FlatKeyReader::cursor(uint32_t startRow) const {
  return encoding_->cursor(startRow);
}

uint32_t FlatKeyReader::rowCount() const {
  return encoding_->rowCount();
}

} // namespace facebook::nimble::index
