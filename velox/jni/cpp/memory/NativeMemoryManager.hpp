/*
* Licensed to the Apache Software Foundation (ASF) under one or more
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

#include <jni.h>

#include "velox/jni/cpp/native/NativeClass.hpp"

namespace facebook::velox::sdk::memory {
class NativeMemoryManager : public jni::NativeClass {
  static std::string CLASS_NAME;

 public:
  NativeMemoryManager() : NativeClass(CLASS_NAME) {}

  void initInternal() override;

  static jlong nativeCreate(JNIEnv* env, jobject obj);

  static jstring nativeMemoryStatics(JNIEnv* env, jobject obj);
};
} // namespace facebook::velox::sdk::memory
