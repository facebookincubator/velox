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

#include "velox/common/file/FileIoTracer.h"

#include <glog/logging.h>

namespace facebook::velox {

std::string IoTag::toString() const {
  if (parent == nullptr) {
    return name;
  }
  return parent->toString() + " -> " + name;
}

size_t IoTag::depth() const {
  size_t d = 1;
  const IoTag* p = parent;
  while (p != nullptr) {
    ++d;
    p = p->parent;
  }
  return d;
}

const IoTag*& threadIoTag() {
  thread_local const IoTag* tag = nullptr;
  return tag;
}

ScopedIoTag::ScopedIoTag(std::string_view name) : tag_(name, threadIoTag()) {
  threadIoTag() = &tag_;
}

ScopedIoTag::~ScopedIoTag() {
  threadIoTag() = tag_.parent;
}

const IoTag& ScopedIoTag::tag() const {
  return tag_;
}

std::string toString(IoType type) {
  switch (type) {
    case IoType::Read:
      return "Read";
    case IoType::AsyncRead:
      return "AsyncRead";
    case IoType::Write:
      return "Write";
    case IoType::AsyncWrite:
      return "AsyncWrite";
  }
  return "Unknown";
}

std::ostream& operator<<(std::ostream& os, IoType type) {
  return os << toString(type);
}

InMemoryFileIoTracer::InMemoryFileIoTracer(std::vector<IoRecord>& records)
    : records_(records) {
  records_.reserve(4096);
}

std::shared_ptr<InMemoryFileIoTracer> InMemoryFileIoTracer::create(
    std::vector<IoRecord>& records) {
  return std::shared_ptr<InMemoryFileIoTracer>(
      new InMemoryFileIoTracer(records));
}

void InMemoryFileIoTracer::record(
    IoType type,
    uint64_t offset,
    uint64_t length) {
  IoRecord record;
  record.type = type;
  record.offset = offset;
  record.length = length;
  const IoTag* tag = threadIoTag();
  record.tag = tag != nullptr ? tag->toString() : "";

  std::lock_guard<std::mutex> lock(mutex_);
  records_.push_back(std::move(record));
}

void InMemoryFileIoTracer::finish() {
  LOG(INFO) << "InMemoryFileIoTracer recorded " << records_.size()
            << " IO operations";
}

std::string IoRecord::toString() const {
  std::string result = velox::toString(type);
  result += " [" + std::to_string(offset) + ", " + std::to_string(length) + "]";
  if (!tag.empty()) {
    result += " " + tag;
  }
  return result;
}

} // namespace facebook::velox
