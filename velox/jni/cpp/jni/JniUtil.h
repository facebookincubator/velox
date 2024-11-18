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
#ifndef VELOX_SDK_JNI_UTIL_H
#define VELOX_SDK_JNI_UTIL_H

#include <glog/logging.h>
#include <jni.h>
#include <string>
#include <vector>
#include "common/base/Exceptions.h"

#include "velox/vector/BaseVector.h"

namespace facebook::velox::sdk {

// copy from implala
class JniUtil {
 public:
  /// Init JniUtil. This should be called prior to any other calls.
  static void Init(JavaVM* g_vm);

  static jmethodID getMethodID(
      JNIEnv* env,
      const std::string& className,
      const std::string& methodName,
      const std::string& sig,
      bool isStatic = false);

  static JNIEnv* GetJNIEnv() {
    int rc = g_vm->GetEnv(reinterpret_cast<void**>(&tls_env_), JNI_VERSION_1_8);
    VELOX_CHECK_EQ(rc, 0, "Unable to get JVM");
    return tls_env_;
  }

 private:
  // Set in Init() once the JVM is initialized.
  static bool jvm_inited_;

  // Thread-local cache of the JNIEnv for this thread.
  static __thread JNIEnv* tls_env_;

  static JavaVM* g_vm;
};
} // namespace facebook::velox::sdk

#endif
