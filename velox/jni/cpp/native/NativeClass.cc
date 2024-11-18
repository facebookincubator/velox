/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NativeClass.hpp"

namespace facebook::velox::sdk::jni {
std::string NativeClass::NATIVE_CLASS_NAME =
    "velox/jni/NativeClass";

NativeClass::NativeClass(const std::string className)
    : className_(className), nativeMethods({}) {}

void NativeClass::init() {
  addNativeMethod(
      "nativeRelease", reinterpret_cast<void*>(nativeRelease), kTypeVoid, NULL);
  initInternal();
  registerNativeMethods();
}

jmethodID NativeClass::initPointMethod(
    std::string className,
    std::string functionName,
    std::string sig) {
  JNIEnv* env = JniUtil::GetJNIEnv();
  //  // Find JniUtil class and create a global ref.
  jclass local_jni_dict_cl = env->FindClass(className.c_str());
  if (local_jni_dict_cl == nullptr) {
    if (env->ExceptionOccurred())
      env->ExceptionDescribe();
  }
  VELOX_CHECK_NOT_NULL(local_jni_dict_cl, "Failed to find JniUtil class.");

  jmethodID nativePointerMethodId_ =
      env->GetMethodID(local_jni_dict_cl, functionName.c_str(), sig.c_str());
  if (nativePointerMethodId_ == NULL) {
    if (env->ExceptionOccurred())
      env->ExceptionDescribe();
  }
  VELOX_CHECK(!env->ExceptionOccurred(), "Failed to find toDictVector method");
  return nativePointerMethodId_;
}

std::string NativeClass::buildJNISignature(
    const char* returnTypeSignature,
    const char* argTypeSignature,
    va_list args) {
  std::string signature = "(";
  const char* tempArgTypeSignature = argTypeSignature;

  while (tempArgTypeSignature != NULL) {
    signature += tempArgTypeSignature;
    tempArgTypeSignature = va_arg(args, const char*);
  }

  signature += ")";
  signature += returnTypeSignature;
  return signature;
}

jlong NativeClass::poniter(jobject java_this) {
  static jmethodID NATIVE_METHOD_HANDLE =
      initPointMethod(NATIVE_CLASS_NAME, "nativePTR", "()J");
  JNIEnv* env = JniUtil::GetJNIEnv();
  return env->CallLongMethod(java_this, NATIVE_METHOD_HANDLE);
}

void NativeClass::addNativeMethod(
    const char* name,
    void* funcPtr,
    const char* returnTypeSignature,
    const char* argTypeSignature,
    ...) {
  va_list args;
  va_start(args, argTypeSignature);
  std::string signature =
      buildJNISignature(returnTypeSignature, argTypeSignature, args);
  va_end(args);
  JNINativeMethod method;
  method.name = strdup(name);
  method.signature = strdup(signature.c_str());
  method.fnPtr = funcPtr;
  nativeMethods.push_back(method);
}

jint NativeClass::registerNativeMethods() {
  JNIEnv* env = JniUtil::GetJNIEnv();
  jclass clazz = env->FindClass(className_.c_str());
  if (clazz == NULL) {
    return JNI_FALSE;
  }

  if (env->RegisterNatives(clazz, &nativeMethods[0], nativeMethods.size()) <
      0) {
    return JNI_FALSE;
  }

  freeNativeMethods();
  return JNI_TRUE;
}

void NativeClass::freeNativeMethods() {
  for (auto& method : nativeMethods) {
    free((void*)method.name);
    free((void*)method.signature);
  }
  nativeMethods.clear();
}

void NativeClass::nativeRelease(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
  static jmethodID NATIVE_METHOD_HANDLE =
      initPointMethod(NATIVE_CLASS_NAME, "releaseHandle", "()V");
  JNIEnv* env = JniUtil::GetJNIEnv();
  delete reinterpret_cast<SharedPtrHandle*>(NativeClass::poniter(obj));
  env->CallLongMethod(obj, NATIVE_METHOD_HANDLE);
  JNI_METHOD_END()
}
} // namespace facebook::velox::sdk::jni
