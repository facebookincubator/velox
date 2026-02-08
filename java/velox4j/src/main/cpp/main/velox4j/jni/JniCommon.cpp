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
#include "velox4j/jni/JniCommon.h"

#include <ClassRegistry.h>
#include <JavaThreadUtils.h>
#include <JniHelpersCommon.h>
#include <fmt/core.h>
#include <glog/logging.h>
#include <jni.h>
#include <velox/common/base/Exceptions.h>
#include <atomic>
#include <cstring>
#include <ostream>
#include <vector>

namespace facebook::velox4j {

const jint kJniVersion = JNI_VERSION_1_8;

std::string jStringToCString(JNIEnv* env, jstring string) {
  const int32_t clen = env->GetStringUTFLength(string);
  const int32_t jlen = env->GetStringLength(string);
  std::string buffer{};
  buffer.resize(clen);
  env->GetStringUTFRegion(string, 0, jlen, &buffer[0]);
  return buffer;
}

void checkException(JNIEnv* env) {
  if (env->ExceptionCheck()) {
    // An exception was thrown from Java. Rethrow it as a C++ exception.
    jthrowable throwable = env->ExceptionOccurred();
    env->ExceptionClear();
    jclass describerClass =
        env->FindClass("com/facebook/velox4j/exception/ExceptionDescriber");
    jmethodID describeMethod = env->GetStaticMethodID(
        describerClass,
        "describe",
        "(Ljava/lang/Throwable;)Ljava/lang/String;");
    std::string description = jStringToCString(
        env,
        (jstring)env->CallStaticObjectMethod(
            describerClass, describeMethod, throwable));
    if (env->ExceptionCheck()) {
      // A new exception was thrown when trying to call Java API to
      // describe the previous exception. We just log this exception,
      // but do not throw it to let the previous exception be reported as the
      // root cause.
      LOG(WARNING) << "Fatal: Uncaught Java exception during calling the Java "
                      "exception describer method! ";
    }
    // Throws the C++ exception.
    VELOX_FAIL(
        "Error during calling Java code from native code: " + description);
  }
}

jclass createGlobalClassReference(JNIEnv* env, const char* className) {
  jclass localClass = env->FindClass(className);
  jclass globalClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);
  return globalClass;
}

jclass createGlobalClassReferenceOrError(JNIEnv* env, const char* className) {
  jclass globalClass = createGlobalClassReference(env, className);
  if (globalClass == nullptr) {
    std::string errorMessage = fmt::format(
        "Unable to create a global class reference for {} ",
        std::string(className));
    VELOX_FAIL(errorMessage);
  }
  return globalClass;
}

JNIEnv* getLocalJNIEnv() {
  static std::atomic<uint32_t> nextThreadId{0};
  if (spotify::jni::JavaThreadUtils::getEnvForCurrentThread() == nullptr) {
    const std::string threadName =
        fmt::format("Velox4J Native Thread {}", nextThreadId++);
    std::vector<char> threadNameCStr(threadName.length() + 1);
    std::strcpy(threadNameCStr.data(), threadName.data());
    JavaVM* vm = spotify::jni::JavaThreadUtils::getJavaVM();
    JNIEnv* env{nullptr};
    JavaVMAttachArgs args;
    args.version = JAVA_VERSION;
    args.name = threadNameCStr.data();
    args.group = nullptr;
    const int result =
        vm->AttachCurrentThreadAsDaemon(reinterpret_cast<void**>(&env), &args);
    if (result != JNI_OK) {
      VELOX_FAIL("Failed to reattach current thread to JVM.");
    }
    return env;
  }
  JNIEnv* env = spotify::jni::JavaThreadUtils::getEnvForCurrentThread();
  VELOX_CHECK(env != nullptr);
  return env;
}

spotify::jni::ClassRegistry* jniClassRegistry() {
  static spotify::jni::ClassRegistry gClasses;
  return &gClasses;
}
} // namespace facebook::velox4j
