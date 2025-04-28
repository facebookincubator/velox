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

#include <JniHelpers.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <velox/vector/ComplexVector.h>
#include "velox4j/connector/ExternalStream.h"

namespace velox4j {
class DownIteratorJniWrapper final : public spotify::jni::JavaClass {
 public:
  explicit DownIteratorJniWrapper(JNIEnv* env) : JavaClass(env) {
    DownIteratorJniWrapper::initialize(env);
  }

  DownIteratorJniWrapper() : JavaClass(){};

  const char* getCanonicalName() const override;

  void initialize(JNIEnv* env) override;

  void mapFields() override;
};

class DownIterator : public ExternalStream {
 public:
  enum class State { AVAILABLE = 0, BLOCKED = 1, FINISHED = 2 };

  // CTOR.
  DownIterator(JNIEnv* env, jobject ref);

  // Delete copy/move CTORs.
  DownIterator(DownIterator&&) = delete;
  DownIterator(const DownIterator&) = delete;
  DownIterator& operator=(const DownIterator&) = delete;
  DownIterator& operator=(DownIterator&&) = delete;

  // DTOR.
  ~DownIterator() override;

  std::optional<facebook::velox::RowVectorPtr> read(
      facebook::velox::ContinueFuture& future) override;

 private:
  State advance();
  void wait();
  facebook::velox::RowVectorPtr get();
  void close();

  jobject ref_;
  std::mutex mutex_;
  std::unique_ptr<folly::IOThreadPoolExecutor> waitExecutor_;
  std::vector<facebook::velox::ContinuePromise> promises_{};
  std::atomic_bool closed_{false};
};

} // namespace velox4j
