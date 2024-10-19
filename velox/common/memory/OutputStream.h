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

#include "velox/common/base/ByteOutputStream.h"

#include <folly/io/IOBuf.h>
#include <memory>

namespace facebook::velox {

class OutputStreamListener {
 public:
  virtual void onWrite(const char* /* s */, std::streamsize /* count */) {}
  virtual ~OutputStreamListener() = default;
};

class OutputStream {
 public:
  explicit OutputStream(OutputStreamListener* listener = nullptr)
      : listener_(listener) {}

  virtual ~OutputStream() = default;

  virtual void write(const char* s, std::streamsize count) = 0;

  virtual std::streampos tellp() const = 0;

  virtual void seekp(std::streampos pos) = 0;

  OutputStreamListener* listener() const {
    return listener_;
  }

 protected:
  OutputStreamListener* listener_;
};

class OStreamOutputStream : public OutputStream {
 public:
  explicit OStreamOutputStream(
      std::ostream* out,
      OutputStreamListener* listener = nullptr)
      : OutputStream(listener), out_(out) {}

  void write(const char* s, std::streamsize count) override {
    out_->write(s, count);
    if (listener_) {
      listener_->onWrite(s, count);
    }
  }

  std::streampos tellp() const override {
    return out_->tellp();
  }

  void seekp(std::streampos pos) override {
    out_->seekp(pos);
  }

 private:
  std::ostream* out_;
};

class IOBufOutputStream : public OutputStream {
 public:
  explicit IOBufOutputStream(
      memory::MemoryPool& pool,
      OutputStreamListener* listener = nullptr,
      int32_t initialSize = memory::AllocationTraits::kPageSize)
      : OutputStream(listener),
        arena_(std::make_shared<StreamArena>(&pool)),
        out_(std::make_unique<ByteOutputStream>(arena_.get())) {
    out_->startWrite(initialSize);
  }

  void write(const char* s, std::streamsize count) override {
    out_->appendStringView(std::string_view(s, count));
    if (listener_) {
      listener_->onWrite(s, count);
    }
  }

  std::streampos tellp() const override;

  void seekp(std::streampos pos) override;

  /// 'releaseFn' is executed on iobuf destruction if not null.
  std::unique_ptr<folly::IOBuf> getIOBuf(
      const std::function<void()>& releaseFn = nullptr);

 private:
  std::shared_ptr<StreamArena> arena_;
  std::unique_ptr<ByteOutputStream> out_;
};

} // namespace facebook::velox
