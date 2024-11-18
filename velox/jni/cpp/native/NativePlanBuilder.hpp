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

#ifndef NATIVEPLANBUILDER_HPP
#define NATIVEPLANBUILDER_HPP
#include <jni.h>

#include "NativeClass.hpp"

namespace facebook::velox::sdk {
class NativePlanBuilder : public jni::NativeClass {
  static std::string CLASS_NAME;

 public:
  NativePlanBuilder() : NativeClass(CLASS_NAME) {}

  void initInternal() override;

  static jlong nativeCreate(JNIEnv* env, jobject obj);

  static void
  nativeProject(JNIEnv* env, jobject obj, jobjectArray projections);

  static jstring nativeJavaScan(JNIEnv* env, jobject obj, jstring schema);

  static void nativeFilter(JNIEnv* env, jobject obj, jstring filter);

  static jstring nativeWindowFunction(
      JNIEnv* env,
      jobject obj,
      jstring functionCall,
      jstring fram,
      jboolean ignoreNullKey);

  static void nativeLimit(JNIEnv* env, jobject obj, jint offset, jint limit);

  static jstring nativeNodeId(JNIEnv* env, jobject obj);

  static jstring nativeBuilder(JNIEnv* env, jobject obj);

  static void nativeTestString(JNIEnv* env, jobject obj, jstring str);

};
} // namespace facebook::velox::sdk

#endif // NATIVEPLANBUILDER_HPP
