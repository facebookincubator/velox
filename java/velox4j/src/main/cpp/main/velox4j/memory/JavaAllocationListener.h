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
#include <atomic>

#include "velox4j/memory/AllocationListener.h"

namespace facebook::velox4j {

// The JNI wrapper used by JavaAllocationListener.
class JavaAllocationListenerJniWrapper final : public spotify::jni::JavaClass {
 public:
  explicit JavaAllocationListenerJniWrapper(JNIEnv* env) : JavaClass(env) {
    JavaAllocationListenerJniWrapper::initialize(env);
  }

  JavaAllocationListenerJniWrapper() : JavaClass(){};

  const char* getCanonicalName() const override;

  void initialize(JNIEnv* env) override;

  void mapFields() override;
};

/// A AllocationListener implementation that is backed by a Java-side
/// allocation listener. The calls to this listener will be redirected to
/// the methods with the same name in Java-side through JNI.
class JavaAllocationListener : public AllocationListener {
 public:
  JavaAllocationListener(JNIEnv* env, jobject ref);

  ~JavaAllocationListener() override;

  void allocationChanged(int64_t diff) override;

  const int64_t currentBytes() const override;

  const int64_t peakBytes() const override;

 private:
  jobject ref_;
  std::atomic_int64_t usedBytes_{0L};
  std::atomic_int64_t peakBytes_{0L};
};
} // namespace facebook::velox4j
