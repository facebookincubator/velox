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

/// A static JNI wrapper that is independent to any JNI sessions.
/// All the JNI methods defined in the static JNI wrapper are globally
/// available without having to create a session first.
class StaticJniWrapper final : public spotify::jni::JavaClass {
 public:
  explicit StaticJniWrapper(JNIEnv* env) : JavaClass(env) {
    StaticJniWrapper::initialize(env);
  }

  const char* getCanonicalName() const override;

  void initialize(JNIEnv* env) override;

  void mapFields() override;
};
} // namespace facebook::velox4j
