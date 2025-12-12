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

#include <stdexcept>

#include "velox4j/jni/JniCommon.h"

namespace facebook::velox4j {

/// A state structure that provides utility APIs in relation
/// to the Java exception types used by Velox4J.
class JniErrorState {
 public:
  virtual ~JniErrorState() = default;

  void ensureInitialized(JNIEnv* env);

  void assertInitialized() const;

  void close();

  jclass runtimeExceptionClass();

  jclass illegalAccessExceptionClass();

  jclass veloxExceptionClass();

 private:
  void initialize(JNIEnv* env);

  jclass ioExceptionClass_{nullptr};
  jclass runtimeExceptionClass_{nullptr};
  jclass unsupportedOperationExceptionClass_{nullptr};
  jclass illegalAccessExceptionClass_{nullptr};
  jclass illegalArgumentExceptionClass_{nullptr};
  jclass veloxExceptionClass_{nullptr};
  JavaVM* vm_{nullptr};
  bool initialized_{false};
  bool closed_{false};
  std::mutex mtx_;
};

JniErrorState* getJniErrorState();

} // namespace facebook::velox4j
