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
#include <jni.h>
#include <jni_md.h>
#include <stdint.h>
#include <cmath>
#include <mutex>
#include <string>

#ifndef JNI_METHOD_START
#define JNI_METHOD_START try {
// macro ended
#endif

#ifndef JNI_METHOD_END
#define JNI_METHOD_END(fallbackExpr)                                  \
  }                                                                   \
  catch (std::exception & e) {                                        \
    env->ThrowNew(                                                    \
        facebook::velox4j::getJniErrorState()->veloxExceptionClass(), \
        e.what());                                                    \
    return fallbackExpr;                                              \
  }
// macro ended
#endif

namespace facebook::velox4j {

/// Checks whether an exception was thrown from Java to C++. If yes, rethrows
/// it as a C++ exception.
void checkException(JNIEnv* env);

std::string jStringToCString(JNIEnv* env, jstring string);

jclass createGlobalClassReference(JNIEnv* env, const char* className);

jclass createGlobalClassReferenceOrError(JNIEnv* env, const char* className);

JNIEnv* getLocalJNIEnv();

spotify::jni::ClassRegistry* jniClassRegistry();

/// Code for implementing safe version of JNI
/// {Get|Release}<PrimitiveType>ArrayElements routines. SafeNativeArray would
/// release the managed array elements automatically during destruction.

#define CONCATENATE(t1, t2, t3) t1##t2##t3

#define DEFINE_PRIMITIVE_ARRAY(                                           \
    PRIM_TYPE, JAVA_TYPE, JNI_NATIVE_TYPE, NATIVE_TYPE, METHOD_VAR)       \
  template <>                                                             \
  struct JniPrimitiveArray<JniPrimitiveArrayType::PRIM_TYPE> {            \
    using JavaType = JAVA_TYPE;                                           \
    using JniNativeType = JNI_NATIVE_TYPE;                                \
    using NativeType = NATIVE_TYPE;                                       \
                                                                          \
    static JniNativeType get(JNIEnv* env, JavaType javaArray) {           \
      return env->CONCATENATE(Get, METHOD_VAR, ArrayElements)(            \
          javaArray, nullptr);                                            \
    }                                                                     \
                                                                          \
    static void                                                           \
    release(JNIEnv* env, JavaType javaArray, JniNativeType nativeArray) { \
      env->CONCATENATE(Release, METHOD_VAR, ArrayElements)(               \
          javaArray, nativeArray, JNI_ABORT);                             \
    }                                                                     \
  };

enum class JniPrimitiveArrayType {
  kBoolean = 0,
  kByte = 1,
  kChar = 2,
  kShort = 3,
  kInt = 4,
  kLong = 5,
  kFloat = 6,
  kDouble = 7
};

template <JniPrimitiveArrayType TYPE>
struct JniPrimitiveArray {};

DEFINE_PRIMITIVE_ARRAY(kBoolean, jbooleanArray, jboolean*, bool*, Boolean)
DEFINE_PRIMITIVE_ARRAY(kByte, jbyteArray, jbyte*, uint8_t*, Byte)
DEFINE_PRIMITIVE_ARRAY(kChar, jcharArray, jchar*, uint16_t*, Char)
DEFINE_PRIMITIVE_ARRAY(kShort, jshortArray, jshort*, int16_t*, Short)
DEFINE_PRIMITIVE_ARRAY(kInt, jintArray, jint*, int32_t*, Int)
DEFINE_PRIMITIVE_ARRAY(kLong, jlongArray, jlong*, int64_t*, Long)
DEFINE_PRIMITIVE_ARRAY(kFloat, jfloatArray, jfloat*, float_t*, Float)
DEFINE_PRIMITIVE_ARRAY(kDouble, jdoubleArray, jdouble*, double_t*, Double)

// A safe native array that handles JNI array releasing in the RAII style.
template <JniPrimitiveArrayType TYPE>
class SafeNativeArray {
  using PrimitiveArray = JniPrimitiveArray<TYPE>;
  using JavaArrayType = typename PrimitiveArray::JavaType;
  using JniNativeArrayType = typename PrimitiveArray::JniNativeType;
  using NativeArrayType = typename PrimitiveArray::NativeType;

 public:
  virtual ~SafeNativeArray() {
    PrimitiveArray::release(env_, javaArray_, nativeArray_);
  }

  SafeNativeArray(const SafeNativeArray&) = delete;
  SafeNativeArray(SafeNativeArray&&) = delete;
  SafeNativeArray& operator=(const SafeNativeArray&) = delete;
  SafeNativeArray& operator=(SafeNativeArray&&) = delete;

  const NativeArrayType elems() const {
    return reinterpret_cast<const NativeArrayType>(nativeArray_);
  }

  const jsize length() const {
    return env_->GetArrayLength(javaArray_);
  }

  static SafeNativeArray<TYPE> get(JNIEnv* env, JavaArrayType javaArray) {
    JniNativeArrayType nativeArray = PrimitiveArray::get(env, javaArray);
    return SafeNativeArray<TYPE>(env, javaArray, nativeArray);
  }

 private:
  SafeNativeArray(
      JNIEnv* env,
      JavaArrayType javaArray,
      JniNativeArrayType nativeArray)
      : env_(env), javaArray_(javaArray), nativeArray_(nativeArray){};

  JNIEnv* env_;
  JavaArrayType javaArray_;
  JniNativeArrayType nativeArray_;
};

#define DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(                             \
    PRIM_TYPE, JAVA_TYPE, METHOD_VAR)                                          \
  inline SafeNativeArray<JniPrimitiveArrayType::PRIM_TYPE> CONCATENATE(        \
      get, METHOD_VAR, ArrayElementsSafe)(JNIEnv * env, JAVA_TYPE array) {     \
    return SafeNativeArray<JniPrimitiveArrayType::PRIM_TYPE>::get(env, array); \
  }

DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kBoolean, jbooleanArray, Boolean)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kByte, jbyteArray, Byte)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kChar, jcharArray, Char)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kShort, jshortArray, Short)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kInt, jintArray, Int)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kLong, jlongArray, Long)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kFloat, jfloatArray, Float)
DEFINE_SAFE_GET_PRIMITIVE_ARRAY_FUNCTIONS(kDouble, jdoubleArray, Double)
} // namespace facebook::velox4j
