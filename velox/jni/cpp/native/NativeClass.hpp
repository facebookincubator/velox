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

#ifndef NATIVECLASS_HPP
#define NATIVECLASS_HPP
#include <jni.h>
#include <string>

#include "velox/jni/cpp/jni/JniErrors.h"
#include "velox/jni/cpp/jni/JniUtil.h"

namespace facebook::velox::sdk::jni {
static std::string GetArrayTypeSignature(const char* baseTypeSignature) {
  auto basic_string = std::string("[") + baseTypeSignature;
  return basic_string;
}

static std::string GetObjectTypeSignature(std::string baseTypeSignature) {
  auto basic_string = std::string("L") + baseTypeSignature +  std::string(";");
  return basic_string;
}

static const char* kTypeBoolean = "Z";
static const char* kTypeByte = "B";
static const char* kTypeChar = "C";
static const char* kTypeShort = "S";
static const char* kTypeInt = "I";
static const char* kTypeLong = "J";
static const char* kTypeFloat = "F";
static const char* kTypeDouble = "D";
static const char* kTypeVoid = "V";
static const char* kTypeString = "Ljava/lang/String;";
static const char* kTypeBooleanArray = "[Z";
static const char* kTypeByteArray = "[B";
static const char* kTypeCharArray = "[C";
static const char* kTypeShortArray = "[S";
static const char* kTypeIntArray = "[I";
static const char* kTypeLongArray = "[J";
static const char* kTypeFloatArray = "[F";
static const char* kTypeDoubleArray = "[D";

class NativeClass {
 public:
  NativeClass(std::string className);

  virtual void initInternal() = 0;

  void init();

  static jmethodID initPointMethod(
      std::string className,
      std::string functionName,
      std::string sig);

  template <typename T>
  static std::shared_ptr<T> as(jobject java_this);

  static jlong poniter(jobject java_this);

  void addNativeMethod(
      const char* name,
      void* funcPtr,
      const char* returnTypeSignature,
      const char* argTypeSignature,
      ...);

  jint registerNativeMethods();

  void freeNativeMethods();

  template <typename T>
  static void RegisterNatives();

  static void nativeRelease(JNIEnv* env, jobject obj);

 private:
  static std::string NATIVE_CLASS_NAME;

  static std::string buildJNISignature(
      const char* returnTypeSignature,
      const char* argTypeSignature,
      va_list args);

  std::string className_;

  std::vector<JNINativeMethod> nativeMethods;
};

template <typename T>
std::shared_ptr<T> NativeClass::as(jobject java_this) {
  jlong handle = poniter(java_this);
  return reinterpret_cast<SharedPtrHandle*>(handle)->as<T>();
}

template <typename T>
void NativeClass::RegisterNatives() {
  JNIEnv* env = JniUtil::GetJNIEnv();
  JNI_METHOD_START
  std::make_shared<T>()->init();
  JNI_METHOD_END();
}

} // namespace facebook::velox::sdk::jni

#endif // NATIVECLASS_HPP
