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

#include <iostream>

#include "NativePlanBuilder.hpp"
#include "velox/core/PlanNode.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/jni/cpp/jni/JniErrors.h"
#include "velox/jni/cpp/jni/JniUtil.h"

#include "velox/parse/TypeResolver.h"

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/jni/cpp/utils/JsonUtils.h"

#include "velox/jni/cpp/expression/ExpressionUtils.h"

namespace facebook::velox::sdk {

using namespace facebook::velox::sdk::jni;
using namespace facebook::velox::sdk::expression;
using namespace facebook::velox::exec::test;

std::string NativePlanBuilder::CLASS_NAME = "velox/jni/NativePlanBuilder";

void NativePlanBuilder::initInternal() {
  addNativeMethod(
      "nativeCreate", reinterpret_cast<void*>(nativeCreate), kTypeLong, NULL);

  addNativeMethod(
      "nativeNodeId", reinterpret_cast<void*>(nativeNodeId), kTypeString, NULL);

  addNativeMethod(
      "nativeBuilder",
      reinterpret_cast<void*>(nativeBuilder),
      kTypeString,
      NULL);

  addNativeMethod(
      "nativeProject",
      reinterpret_cast<void*>(nativeProject),
      kTypeVoid,
      GetArrayTypeSignature(kTypeString).c_str(),
      NULL);

  addNativeMethod(
      "nativeJavaScan",
      reinterpret_cast<void*>(nativeJavaScan),
      kTypeVoid,
      kTypeString,
      NULL);

  addNativeMethod(
      "nativeFilter",
      reinterpret_cast<void*>(nativeFilter),
      kTypeVoid,
      kTypeString,
      NULL);

  addNativeMethod(
      "nativeLimit",
      reinterpret_cast<void*>(nativeLimit),
      kTypeVoid,
      kTypeInt,
      kTypeInt,
      NULL);
}

jlong NativePlanBuilder::nativeCreate(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
  SharedPtrHandle* handle = new SharedPtrHandle{std::make_shared<PlanBuilder>(
      memory::MemoryManager::get()->planMemoryPool().get())};
  return reinterpret_cast<long>(handle);
  JNI_METHOD_END(-1)
}

void NativePlanBuilder::nativeProject(
    JNIEnv* env,
    jobject obj,
    jobjectArray projections) {
  JNI_METHOD_START
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  builder->project(ConvertJStringArrayToVector(env, projections));
  JNI_METHOD_END()
}

jstring
NativePlanBuilder::nativeJavaScan(JNIEnv* env, jobject obj, jstring schema) {
  JNI_METHOD_START
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  const std::string schemaStr = jStringToCString(env, schema);
  const RowTypePtr rowType = asRowType(ExprUtils::toVeloxType(schemaStr));
  builder->tableScan(rowType);
  core::PlanNodeId scanNodeId;
  builder->capturePlanNodeId(scanNodeId);
  return env->NewStringUTF(scanNodeId.c_str());
  JNI_METHOD_END(nullptr)
}

void NativePlanBuilder::nativeFilter(JNIEnv* env, jobject obj, jstring filter) {
  JNI_METHOD_START
  const std::string filterStr = jStringToCString(env, filter);
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  auto previousHook = core::Expressions::getResolverHook();
  parse::registerTypeResolver();
  builder->filter(filterStr);
  core::Expressions::setTypeResolverHook(previousHook);
  JNI_METHOD_END()
}
jstring NativePlanBuilder::nativeWindowFunction(
    JNIEnv* env,
    jobject obj,
    jstring functionCall,
    jstring fram,
    jboolean ignoreNullKey) {
  JNI_METHOD_START
  auto typePtr = ISerializable::deserialize<core::ITypedExpr>(
      folly::parseJson(
          jStringToCString(env, functionCall), getSerializationOptions()),
      sdk::memory::MemoryManager::get()->planMemoryPool().get());
  std::shared_ptr<const core::CallTypedExpr> callExpr =
      std::dynamic_pointer_cast<const core::CallTypedExpr>(typePtr);
  auto frame = core::WindowNode::Frame::deserialize(
      folly::parseJson(jStringToCString(env, fram), getSerializationOptions()));
  bool ignoreNull = ignoreNullKey;
  return env->NewStringUTF(
      folly::toJson(
          core::WindowNode::Function{callExpr, frame, ignoreNull}.serialize())
          .c_str());
  JNI_METHOD_END(env->NewStringUTF(""))
}

void NativePlanBuilder::nativeLimit(
    JNIEnv* env,
    jobject obj,
    jint offset,
    jint limit) {
  JNI_METHOD_START
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  builder->limit(offset, limit, false);
  JNI_METHOD_END()
}

jstring NativePlanBuilder::nativeNodeId(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  return env->NewStringUTF(builder->planNode()->id().c_str());
  JNI_METHOD_END(nullptr)
}

jstring NativePlanBuilder::nativeBuilder(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
  const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
  return env->NewStringUTF(
      utils::JsonUtils::toSortedJson(builder->planNode()->serialize()).c_str());
  JNI_METHOD_END(nullptr)
}

void NativePlanBuilder::nativeTestString(
    JNIEnv* env,
    jobject obj,
    jstring str) {
  JNI_METHOD_START
  std::string s = jStringToCString(env, str);
  std::cout << "String " << s.length() << std::endl;
  JNI_METHOD_END()
}

} // namespace facebook::velox::sdk
