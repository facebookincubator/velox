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
  //
  // static void nativeExpand(
  //     JNIEnv* env,
  //     jobject obj,
  //     jobjectArray projects,
  //     jobjectArray alias);
  //
  // static void nativeShuffledHashJoin(
  //     JNIEnv* env,
  //     jobject obj,
  //     jstring joinType,
  //     jboolean nullAware,
  //     jobjectArray leftKeys,
  //     jobjectArray rightKeys,
  //     jstring condition,
  //     jstring plan,
  //     jstring output);
  //
  //   static void nativeMergeJoin(
  //     JNIEnv* env,
  //     jobject obj,
  //     jstring joinType,
  //     jobjectArray leftKeys,
  //     jobjectArray rightKeys,
  //     jstring condition,
  //     jstring plan,
  //     jstring output);
  //
  // static void nativeAggregation(
  //     JNIEnv* env,
  //     jobject obj,
  //     jstring step,
  //     jobjectArray groupings,
  //     jobjectArray aggNames,
  //     jobjectArray aggs,
  //     jboolean ignoreNullKey);

  static jstring nativeWindowFunction(
      JNIEnv* env,
      jobject obj,
      jstring functionCall,
      jstring fram,
      jboolean ignoreNullKey);
  //
  // static void nativeWindow(
  //     JNIEnv* env,
  //     jobject obj,
  //     jobjectArray partitionKeys,
  //     jobjectArray sortingKeys,
  //     jobjectArray sortingOrders,
  //     jobjectArray windowColumnNames,
  //     jobjectArray windowFunctions,
  //     jboolean ignoreNullKey);
  //
  // static void nativeSort(
  //     JNIEnv* env,
  //     jobject obj,
  //     jobjectArray sortingKeys,
  //     jobjectArray sortingOrders,
  //     jboolean isPartial);
  //
  // static void nativeUnnest(
  //     JNIEnv* env,
  //     jobject obj,
  //     jobjectArray replicateVariables,
  //     jobjectArray unnestVariables,
  //     jobjectArray unnestNames,
  //     jstring ordinalityName);

  static void nativeLimit(JNIEnv* env, jobject obj, jint offset, jint limit);
  // static void nativePartitionedOutput(
  //     JNIEnv* env,
  //     jobject obj,
  //     jobjectArray jKeys,
  //     jint partitions);

  static jstring nativeNodeId(JNIEnv* env, jobject obj);

  static jstring nativeBuilder(JNIEnv* env, jobject obj);

  static void nativeTestString(JNIEnv* env, jobject obj, jstring str);

};
} // namespace facebook::velox::sdk

#endif // NATIVEPLANBUILDER_HPP
