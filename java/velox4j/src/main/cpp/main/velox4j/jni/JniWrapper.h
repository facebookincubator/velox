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

#include <JavaClass.h>
#include <JniHelpers.h>
#include <jni.h>

namespace facebook::velox4j {

/// A dynamic JniWrapper that includes the JNI methods that are session-aware.
/// Which means, the sanity of these methods usually rely on certain objects
/// that were stored in the current session. For example, an API that turns
/// a Velox vector into another, then returns it to Java - this method will read
/// and write objects from and to the current JNI session storage. So the
/// method will be defined in the (dynamic) JniWrapper.
class JniWrapper final : public spotify::jni::JavaClass {
 public:
  explicit JniWrapper(JNIEnv* env) : JavaClass(env) {
    JniWrapper::initialize(env);
  }

  const char* getCanonicalName() const override;

  void initialize(JNIEnv* env) override;

  void mapFields() override;
};
} // namespace facebook::velox4j
