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

#include "velox/common/file/File.h"

#include <algorithm>
#include <cstring>

namespace facebook::velox {

void IoStats::addCounter(const std::string& name, RuntimeCounter counter) {
  auto locked = stats_.wlock();
  auto it = locked->find(name);
  if (it == locked->end()) {
    auto [ptr, inserted] = locked->emplace(name, RuntimeMetric(counter.unit));
    VELOX_CHECK(inserted);
    ptr->second.addValue(counter.value);
  } else {
    VELOX_CHECK_EQ(it->second.unit, counter.unit);
    it->second.addValue(counter.value);
  }
}

void IoStats::merge(const IoStats& other) {
  auto otherStats = other.stats();
  auto locked = stats_.wlock();
  for (const auto& [name, metric] : otherStats) {
    auto it = locked->find(name);
    if (it == locked->end()) {
      locked->emplace(name, metric);
    } else {
      it->second.merge(metric);
    }
  }
}

folly::F14FastMap<std::string, RuntimeMetric> IoStats::stats() const {
  return stats_.copy();
}

std::string ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  std::string buf;
  buf.resize(length);
  auto res = pread(offset, length, buf.data(), context);
  buf.resize(res.size());
  return buf;
}

uint64_t ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  auto fileSize = size();
  uint64_t numRead = 0;
  if (offset >= fileSize) {
    return 0;
  }
  for (auto& range : buffers) {
    auto copySize = std::min<size_t>(range.size(), fileSize - offset);
    // NOTE: skip the gap in case of coalesce io.
    if (range.data() != nullptr) {
      pread(offset, copySize, range.data(), context);
    }
    offset += copySize;
    numRead += copySize;
  }
  return numRead;
}

uint64_t ReadFile::preadv(
    folly::Range<const common::Region*> regions,
    folly::Range<folly::IOBuf*> iobufs,
    const FileIoContext& context) const {
  VELOX_CHECK_EQ(regions.size(), iobufs.size());
  uint64_t length = 0;
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    auto& output = iobufs[i];
    output = folly::IOBuf(folly::IOBuf::CREATE, region.length);
    pread(region.offset, region.length, output.writableData(), context);
    output.append(region.length);
    length += region.length;
  }
  return length;
}

uint64_t ReadFile::preadv(
    folly::Range<const common::Region*> regions,
    folly::Range<const folly::Range<char*>*> buffers,
    const FileIoContext& context) const {
  VELOX_CHECK_EQ(
      regions.size(), buffers.size(), "preadv requires one buffer per region");
  uint64_t length = 0;
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    const auto& buffer = buffers[i];
    VELOX_CHECK_NOT_NULL(buffer.data(), "preadv buffer must not be null");
    VELOX_CHECK_EQ(
        buffer.size(),
        region.length,
        "preadv destination buffer length must match region length");
    pread(region.offset, region.length, buffer.data(), context);
    length += region.length;
  }
  return length;
}

std::string_view InMemoryReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileIoContext& context) const {
  bytesRead_ += length;
  memcpy(buf, file_.data() + offset, length);
  return {static_cast<char*>(buf), length};
}

std::string InMemoryReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  bytesRead_ += length;
  return std::string(file_.data() + offset, length);
}

void InMemoryWriteFile::append(std::string_view data) {
  file_->append(data);
}

void InMemoryWriteFile::append(std::unique_ptr<folly::IOBuf> data) {
  for (auto rangeIter = data->begin(); rangeIter != data->end(); ++rangeIter) {
    file_->append(
        reinterpret_cast<const char*>(rangeIter->data()), rangeIter->size());
  }
}

uint64_t InMemoryWriteFile::size() const {
  return file_->size();
}

} // namespace facebook::velox
