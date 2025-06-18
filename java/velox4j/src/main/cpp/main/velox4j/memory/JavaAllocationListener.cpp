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
#include "velox4j/memory/JavaAllocationListener.h"

#include <glog/logging.h>

#include "velox4j/jni/JniCommon.h"

namespace facebook::velox4j {

namespace {

const char* kClassName = "com/facebook/velox4j/memory/AllocationListener";

} // namespace

const char* JavaAllocationListenerJniWrapper::getCanonicalName() const {
  return kClassName;
}

void JavaAllocationListenerJniWrapper::initialize(JNIEnv* env) {
  JavaClass::setClass(env);

  cacheMethod(env, "allocationChanged", kTypeVoid, kTypeLong, nullptr);

  registerNativeMethods(env);
}

void JavaAllocationListenerJniWrapper::mapFields() {
  // No fields to map.
}

JavaAllocationListener::JavaAllocationListener(JNIEnv* env, jobject ref) {
  ref_ = env->NewGlobalRef(ref);
}

JavaAllocationListener::~JavaAllocationListener() {
  try {
    getLocalJNIEnv()->DeleteGlobalRef(ref_);
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Unable to destroy the global reference to the Java side "
                    "allocation listener: "
                 << ex.what();
  }
}

void JavaAllocationListener::allocationChanged(int64_t diff) {
  static const auto* clazz = jniClassRegistry()->get(kClassName);
  static jmethodID methodId = clazz->getMethod("allocationChanged");
  if (diff == 0) {
    return;
  }
  JNIEnv* env = getLocalJNIEnv();
  env->CallLongMethod(ref_, methodId, diff);
  usedBytes_ += diff;
  while (true) {
    int64_t savedPeakBytes = peakBytes_;
    if (usedBytes_ <= savedPeakBytes) {
      break;
    }
    // usedBytes_ > savedPeakBytes, update peak
    if (peakBytes_.compare_exchange_weak(savedPeakBytes, usedBytes_)) {
      break;
    }
  }
}

const int64_t JavaAllocationListener::currentBytes() const {
  return usedBytes_;
}

const int64_t JavaAllocationListener::peakBytes() const {
  return peakBytes_;
}
} // namespace facebook::velox4j
