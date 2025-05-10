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

namespace velox4j {
class JniWrapper final : public spotify::jni::JavaClass {
 public:
  explicit JniWrapper(JNIEnv* env) : JavaClass(env) {
    JniWrapper::initialize(env);
  }

  const char* getCanonicalName() const override;

  void initialize(JNIEnv* env) override;

  void mapFields() override;
};
} // namespace velox4j
