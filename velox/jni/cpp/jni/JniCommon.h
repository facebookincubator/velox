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

#pragma once

#include <jni.h>

#include <atomic>
#include <iostream>
#include <memory>
#include "common/base/Exceptions.h"

static jint jniVersion = JNI_VERSION_1_8;

#define JNI_METHOD_SIG(name, signature) \
  (JNINativeMethod{                     \
      const_cast<char*>(#name),         \
      const_cast<char*>(#signature),    \
      reinterpret_cast<void*>(name)})

struct SharedPtrHandle {
  static std::atomic<int> global_instance_count; // 全局实例计数器，原子操作

  std::shared_ptr<void> plan_node;

  SharedPtrHandle() {
#ifdef VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
    // 默认构造函数，增加全局实例计数器
    global_instance_count.fetch_add(1, std::memory_order_relaxed);
    print_instance_count("Default constructor");
#endif VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
  }

  template <typename T>
  SharedPtrHandle(std::shared_ptr<T> node)
      : plan_node(std::static_pointer_cast<void>(node)) {
#ifdef VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
    // 构造函数，增加全局实例计数器
    global_instance_count.fetch_add(1, std::memory_order_relaxed);
    print_instance_count("Constructor with node");
#endif VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
  }

  // 拷贝构造函数和移动构造函数也需要增加计数器
  SharedPtrHandle(const SharedPtrHandle&) {
#ifdef VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
    global_instance_count.fetch_add(1, std::memory_order_relaxed);
    print_instance_count("Copy constructor");
#endif VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
  }

  SharedPtrHandle(SharedPtrHandle&&) noexcept {
#ifdef VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
    global_instance_count.fetch_add(1, std::memory_order_relaxed);
    print_instance_count("Move constructor");
#endif VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
  }

  // 拷贝赋值操作符和移动赋值操作符不需要修改计数器
  SharedPtrHandle& operator=(const SharedPtrHandle&) = default;
  SharedPtrHandle& operator=(SharedPtrHandle&&) noexcept = default;

  ~SharedPtrHandle() {
#ifdef VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
    // 析构函数，减少全局实例计数器
    global_instance_count.fetch_sub(1, std::memory_order_relaxed);
    print_instance_count("Destructor");
#endif VELOX_ENABLE_JNI_BINDINGS_REF_DEBUG
  }

  template <typename T>
  std::shared_ptr<T> as() const {
    VELOX_CHECK_NOT_NULL(plan_node, "SharedPtrHandle does not hold an object.");
    auto casted_ptr = std::static_pointer_cast<T>(plan_node);
    VELOX_CHECK_NOT_NULL(casted_ptr, "Failed to cast to the requested type.");
    return casted_ptr;
  }

  static void print_instance_count(const char* message) {
    // 打印当前全局实例计数器的值
    std::cout << message << ": SharedPtrHandle instances count is "
              << global_instance_count.load(std::memory_order_relaxed)
              << std::endl;
  }
};

static inline jclass createGlobalClassReference(
    JNIEnv* env,
    const char* className) {
  jclass localClass = env->FindClass(className);
  jclass globalClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);
  return globalClass;
}

static inline jmethodID
getMethodId(JNIEnv* env, jclass thisClass, const char* name, const char* sig) {
  jmethodID ret = env->GetMethodID(thisClass, name, sig);
  return ret;
}

static inline jmethodID getStaticMethodId(
    JNIEnv* env,
    jclass thisClass,
    const char* name,
    const char* sig) {
  jmethodID ret = env->GetStaticMethodID(thisClass, name, sig);
  return ret;
}

static std::string jbyteArrayToString(JNIEnv* env, jbyteArray array) {
  // 获取数组长度
  jsize length = env->GetArrayLength(array);

  // 如果数组为空，返回空的 std::string
  if (length == 0) {
    return std::string();
  }

  // 获取数组元素
  jbyte* bytes = env->GetByteArrayElements(array, nullptr);

  // 创建 std::string 对象
  std::string str(reinterpret_cast<char*>(bytes), length);

  // 释放数组元素
  env->ReleaseByteArrayElements(array, bytes, JNI_ABORT);

  return str;
}

static inline std::string jStringToCString(JNIEnv* env, jstring jStr) {
  if (!jStr) {
    return "";
  }
  // 将jstring转换为UTF-8格式的C字符串
  const char* chars = env->GetStringUTFChars(jStr, nullptr);
  // 使用C字符串创建std::string实例
  std::string ret(chars);
  // 通知JVM不再需要这个C字符串了
  env->ReleaseStringUTFChars(jStr, chars);
  return ret;
}

static inline std::vector<std::string> ConvertJStringArrayToVector(
    JNIEnv* env,
    jobjectArray jStringArray) {
  if (!jStringArray) {
    return {};
  }
  // 获取数组长度
  jsize arrayLength = env->GetArrayLength(jStringArray);

  // 创建一个std::vector<std::string>，预留足够的空间
  std::vector<std::string> result;
  result.reserve(arrayLength);

  // 遍历Java字符串数组
  for (jsize i = 0; i < arrayLength; i++) {
    // 获取Java字符串
    jstring jStr = (jstring)env->GetObjectArrayElement(jStringArray, i);

    // 检查是否正常获取到Java字符串
    if (jStr) {
      // 将Java字符串转换为C++字符串
      const char* rawStr = env->GetStringUTFChars(jStr, nullptr);
      std::string str(rawStr);

      // 释放Java字符串内存
      env->ReleaseStringUTFChars(jStr, rawStr);

      // 将C++字符串添加到vector中
      result.push_back(str);

      // 删除局部引用，防止内存泄漏
      env->DeleteLocalRef(jStr);
    }
  }

  return result;
}




