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
#include "velox4j/jni/JniError.h"

#include <velox/common/base/Exceptions.h>

namespace facebook::velox4j {

void JniErrorState::ensureInitialized(JNIEnv* env) {
  std::lock_guard<std::mutex> lockGuard(mtx_);
  if (initialized_) {
    return;
  }
  initialize(env);
  initialized_ = true;
}

void JniErrorState::assertInitialized() const {
  if (!initialized_) {
    VELOX_FAIL(
        "Fatal: JniErrorState::Initialize(...) was not called before "
        "using the utility");
  }
}

jclass JniErrorState::runtimeExceptionClass() {
  assertInitialized();
  return runtimeExceptionClass_;
}

jclass JniErrorState::illegalAccessExceptionClass() {
  assertInitialized();
  return illegalAccessExceptionClass_;
}

jclass JniErrorState::veloxExceptionClass() {
  assertInitialized();
  return veloxExceptionClass_;
}

void JniErrorState::initialize(JNIEnv* env) {
  veloxExceptionClass_ = createGlobalClassReferenceOrError(
      env, "Lcom/facebook/velox4j/exception/VeloxException;");
  ioExceptionClass_ =
      createGlobalClassReferenceOrError(env, "Ljava/io/IOException;");
  runtimeExceptionClass_ =
      createGlobalClassReferenceOrError(env, "Ljava/lang/RuntimeException;");
  unsupportedOperationExceptionClass_ = createGlobalClassReferenceOrError(
      env, "Ljava/lang/UnsupportedOperationException;");
  illegalAccessExceptionClass_ = createGlobalClassReferenceOrError(
      env, "Ljava/lang/IllegalAccessException;");
  illegalArgumentExceptionClass_ = createGlobalClassReferenceOrError(
      env, "Ljava/lang/IllegalArgumentException;");
  JavaVM* vm;
  if (env->GetJavaVM(&vm) != JNI_OK) {
    VELOX_FAIL("Unable to get JavaVM instance");
  }
  vm_ = vm;
}

void JniErrorState::close() {
  std::lock_guard<std::mutex> lockGuard(mtx_);
  if (closed_) {
    return;
  }
  JNIEnv* env = getLocalJNIEnv();
  env->DeleteGlobalRef(veloxExceptionClass_);
  env->DeleteGlobalRef(ioExceptionClass_);
  env->DeleteGlobalRef(runtimeExceptionClass_);
  env->DeleteGlobalRef(unsupportedOperationExceptionClass_);
  env->DeleteGlobalRef(illegalAccessExceptionClass_);
  env->DeleteGlobalRef(illegalArgumentExceptionClass_);
  closed_ = true;
}

JniErrorState* getJniErrorState() {
  static JniErrorState jniErrorState;
  return &jniErrorState;
}
} // namespace facebook::velox4j
