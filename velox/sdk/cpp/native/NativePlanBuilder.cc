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

#include <core/PlanNode.h>
#include <exec/AggregateFunctionRegistry.h>
#include "sdk/cpp/jni/JniErrors.h"
#include "sdk/cpp/jni/JniUtil.h"

#include <iostream>
#include "NativePlanBuilder.hpp"

#include <exec/tests/utils/PlanBuilder.h>
#include <sdk/cpp/utils/JsonUtils.h>

#include "sdk/cpp/expression/ExpressionUtils.hpp"

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

  // addNativeMethod(
  //     "nativeAggregation",
  //     reinterpret_cast<void*>(nativeAggregation),
  //     kTypeVoid,
  //     kTypeString,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeBoolean,
  //     NULL);
  //
  // addNativeMethod(
  //     "nativeShuffledHashJoin",
  //     reinterpret_cast<void*>(nativeShuffledHashJoin),
  //     kTypeVoid,
  //     kTypeString,
  //     kTypeBoolean,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeString,
  //     kTypeString,
  //     kTypeString,
  //     NULL);
  // addNativeMethod(
  //     "nativeMergeJoin",
  //     reinterpret_cast<void*>(nativeMergeJoin),
  //     kTypeVoid,
  //     kTypeString,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeString,
  //     kTypeString,
  //     kTypeString,
  //     NULL);
  // addNativeMethod(
  //     "nativeExpand",
  //     reinterpret_cast<void*>(nativeExpand),
  //     kTypeVoid,
  //     "[[Ljava/lang/String;",
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     NULL);
  // addNativeMethod(
  //     "nativeTestString",
  //     reinterpret_cast<void*>(nativeTestString),
  //     kTypeVoid,
  //     kTypeString,
  //     NULL);

  // addNativeMethod(
  //     "nativeWindowFunction",
  //     reinterpret_cast<void*>(nativeWindowFunction),
  //     kTypeString,
  //     kTypeString,
  //     kTypeString,
  //     kTypeBoolean,
  //     NULL);

  // addNativeMethod(
  //     "nativeWindow",
  //     reinterpret_cast<void*>(nativeWindow),
  //     kTypeVoid,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeBoolean,
  //     NULL);
  //
  // addNativeMethod(
  //     "nativeSort",
  //     reinterpret_cast<void*>(nativeSort),
  //     kTypeVoid,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeBoolean,
  //     NULL);
  //
  // addNativeMethod(
  //     "nativeUnnest",
  //     reinterpret_cast<void*>(nativeUnnest),
  //     kTypeVoid,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeString,
  //     NULL);

  addNativeMethod(
      "nativeLimit",
      reinterpret_cast<void*>(nativeLimit),
      kTypeVoid,
      kTypeInt,
      kTypeInt,
      NULL);
  // addNativeMethod(
  //     "nativePartitionedOutput",
  //     reinterpret_cast<void*>(nativePartitionedOutput),
  //     kTypeVoid,
  //     GetArrayTypeSignature(kTypeString).c_str(),
  //     kTypeInt,
  //     NULL);
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
//
// void NativePlanBuilder::nativeExpand(
//     JNIEnv* env,
//     jobject obj,
//     jobjectArray projects,
//     jobjectArray alias) {
//   JNI_METHOD_START
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   jsize arrayLength = env->GetArrayLength(projects);
//
//   // 创建一个std::vector<std::string>，预留足够的空间
//   std::vector<std::string> result;
//   result.reserve(arrayLength);
//
//   std::vector<std::vector<core::TypedExprPtr>> cProjects = {};
//   // 遍历Java字符串数组
//   for (jsize i = 0; i < arrayLength; i++) {
//     // 获取Java字符串
//     jobjectArray arr = (jobjectArray)env->GetObjectArrayElement(projects, i);
//     // 检查是否正常获取到Java字符串
//     if (arr) {
//       std::vector<core::TypedExprPtr> fields =
//           ExprUtils::asISerializableVector<core::ITypedExpr>(
//               env, arr);
//       cProjects.push_back(fields);
//       // 删除局部引用，防止内存泄漏
//       env->DeleteLocalRef(arr);
//     }
//   }
//   std::vector<std::string> cAlias = ConvertJStringArrayToVector(env, alias);
//   builder->expand(cProjects, cAlias);
//   JNI_METHOD_END()
// }
//
// void NativePlanBuilder::nativeShuffledHashJoin(
//     JNIEnv* env,
//     jobject obj,
//     jstring joinType,
//     jboolean nullAware,
//     jobjectArray leftKeys,
//     jobjectArray rightKeys,
//     jstring condition,
//     jstring plan,
//     jstring output) {
//   JNI_METHOD_START
//   core::TypedExprPtr filter = nullptr;
//   if (condition != NULL) {
//     filter = ExprUtils::asISerializable<core::ITypedExpr>(
//         env, condition);
//   }
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   std::shared_ptr<const core::PlanNode> planPtr = nullptr;
//   if (plan) {
//     std::string planJson = jStringToCString(env, plan);
//     planPtr = ISerializable::deserialize<core::PlanNode>(
//         folly::parseJson(planJson, getSerializationOptions()),
//         sdk::memory::MemoryManager::get()->planMemoryPool().get());
//   }
//   builder->join(
//       core::joinTypeFromName(jStringToCString(env, joinType)),
//       nullAware,
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, leftKeys),
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, rightKeys),
//       filter,
//       planPtr,
//       asRowType(ExprUtils::toVeloxType(
//           jStringToCString(env, output))));
//
//   JNI_METHOD_END()
// }
// void NativePlanBuilder::nativeMergeJoin(
//     JNIEnv* env,
//     jobject obj,
//     jstring joinType,
//     jobjectArray leftKeys,
//     jobjectArray rightKeys,
//     jstring condition,
//     jstring plan,
//     jstring output) {
//   JNI_METHOD_START
//   core::TypedExprPtr filter = nullptr;
//   if (condition != NULL) {
//     filter = ExprUtils::asISerializable<core::ITypedExpr>(
//         env, condition);
//   }
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   std::shared_ptr<const core::PlanNode> planPtr = nullptr;
//   if (plan) {
//     std::string planJson = jStringToCString(env, plan);
//     planPtr = ISerializable::deserialize<core::PlanNode>(
//         folly::parseJson(planJson, getSerializationOptions()),
//         sdk::memory::MemoryManager::get()->planMemoryPool().get());
//   }
//   builder->mergeJoin(
//       core::joinTypeFromName(jStringToCString(env, joinType)),
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, leftKeys),
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, rightKeys),
//       filter,
//       planPtr,
//       asRowType(ExprUtils::toVeloxType(
//           jStringToCString(env, output))));
//   JNI_METHOD_END()
// }
//
// void NativePlanBuilder::nativeAggregation(
//     JNIEnv* env,
//     jobject obj,
//     jstring step,
//     jobjectArray groupings,
//     jobjectArray aggNames,
//     jobjectArray aggs,
//     jboolean ignoreNullKey) {
//   JNI_METHOD_START
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   std::vector<core::FieldAccessTypedExprPtr> groupFields =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, groupings);
//   std::vector<core::AggregationNode::Aggregate> aggregates = {};
//   for (auto agg : ConvertJStringArrayToVector(env, aggs)) {
//     aggregates.push_back(core::AggregationNode::Aggregate::deserialize(
//         folly::parseJson(agg, getSerializationOptions()),
//         sdk::memory::MemoryManager::get()->planMemoryPool().get()));
//   }
//
//   builder->aggregation(
//       core::AggregationNode::stepFromName(jStringToCString(env, step)),
//       groupFields,
//       {},
//       ConvertJStringArrayToVector(env, aggNames),
//       aggregates,
//       ignoreNullKey);
//
//   JNI_METHOD_END()
// }
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
// void NativePlanBuilder::nativeWindow(
//     JNIEnv* env,
//     jobject obj,
//     jobjectArray partitionKeys,
//     jobjectArray sortingKeys,
//     jobjectArray sortingOrders,
//     jobjectArray windowColumnNames,
//     jobjectArray windowFunctions,
//     jboolean ignoreNullKey) {
//   JNI_METHOD_START
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//
//   std::vector<core::FieldAccessTypedExprPtr> partitionsKeyExprs =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, partitionKeys);
//   std::vector<core::FieldAccessTypedExprPtr> sortingKeyExprs =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, sortingKeys);
//
//   std::vector<core::SortOrder> sortingOrderExprs =
//       ExprUtils::deserializeArray<core::SortOrder>(
//           env, sortingOrders);
//
//   std::vector<core::WindowNode::Function> windowFunctionExprs =
//       ExprUtils::deserializeArray<
//           core::WindowNode::Function>(env, windowFunctions);
//
//   bool cIgnoreNullKey = ignoreNullKey;
//   builder->window(
//       partitionsKeyExprs,
//       sortingKeyExprs,
//       sortingOrderExprs,
//       std::move(ConvertJStringArrayToVector(env, windowColumnNames)),
//       windowFunctionExprs,
//       cIgnoreNullKey);
//   JNI_METHOD_END()
// }
//
// void NativePlanBuilder::nativeSort(
//     JNIEnv* env,
//     jobject obj,
//     jobjectArray sortingKeys,
//     jobjectArray sortingOrders,
//     jboolean isPartial) {
//   JNI_METHOD_START
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   std::vector<core::FieldAccessTypedExprPtr> sortingKeyExprs =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, sortingKeys);
//   std::vector<core::SortOrder> sortingOrderExprs =
//       ExprUtils::deserializeArray<core::SortOrder>(
//           env, sortingOrders);
//   builder->sort(sortingKeyExprs, sortingOrderExprs, isPartial);
//   JNI_METHOD_END()
// }
//
// void NativePlanBuilder::nativeUnnest(
//     JNIEnv* env,
//     jobject obj,
//     jobjectArray replicateVariables,
//     jobjectArray unnestVariables,
//     jobjectArray unnestNames,
//     jstring ordinalityName) {
//   JNI_METHOD_START
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   std::vector<core::FieldAccessTypedExprPtr> replicateVariablesExprs =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, replicateVariables);
//   std::vector<core::FieldAccessTypedExprPtr> unnestVariableExprs =
//       ExprUtils::asISerializableVector<
//           core::FieldAccessTypedExpr>(env, unnestVariables);
//   std::vector<std::string> cUnnestNames =
//       ConvertJStringArrayToVector(env, unnestNames);
//   std::optional<std::string> ordinalColumn = std::nullopt;
//   if (ordinalityName) {
//     ordinalColumn = jStringToCString(env, ordinalityName);
//   }
//   builder->unnest(
//       replicateVariablesExprs,
//       unnestVariableExprs,
//       cUnnestNames,
//       ordinalColumn);
//   JNI_METHOD_END()
// }

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
//
// void NativePlanBuilder::nativePartitionedOutput(
//     JNIEnv* env,
//     jobject obj,
//     jobjectArray jKeys,
//     jint numPartitions) {
//   JNI_METHOD_START
//   std::vector<core::TypedExprPtr> keys =
//       ExprUtils::asISerializableVector<core::ITypedExpr>(
//           env, jKeys);
//   const std::shared_ptr<PlanBuilder> builder = as<PlanBuilder>(obj);
//   builder->partitionedOutput(keys, numPartitions);
//   JNI_METHOD_END()
// }

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
  JNI_METHOD_END(env->NewStringUTF("ERROR"))
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