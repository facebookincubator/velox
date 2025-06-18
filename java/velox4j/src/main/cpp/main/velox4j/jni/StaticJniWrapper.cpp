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
#include "velox4j/jni/StaticJniWrapper.h"

#include <JavaString.h>
#include <folly/json/json.h>
#include <jni_md.h>
#include <velox/common/encode/Base64.h>
#include <velox/common/serialization/Serializable.h>
#include <velox/connectors/Connector.h>
#include <velox/exec/TableWriter.h>
#include <velox/type/Type.h>
#include <velox/type/Variant.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/SelectivityVector.h>
#include <velox/vector/TypeAliases.h>
#include <velox/vector/VectorEncoding.h>
#include <velox/vector/VectorSaver.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "JniCommon.h"
#include "JniError.h"
#include "JniTypes.h"
#include "velox4j/arrow/Arrow.h"
#include "velox4j/conf/Config.h"
#include "velox4j/init/Init.h"
#include "velox4j/iterator/BlockingQueue.h"
#include "velox4j/iterator/UpIterator.h"
#include "velox4j/lifecycle/ObjectStore.h"
#include "velox4j/lifecycle/Session.h"
#include "velox4j/memory/AllocationListener.h"
#include "velox4j/memory/JavaAllocationListener.h"
#include "velox4j/memory/MemoryManager.h"
#include "velox4j/query/QueryExecutor.h"

namespace facebook::velox4j {
using namespace facebook::velox;

namespace {
const char* kClassName = "com/facebook/velox4j/jni/StaticJniWrapper";

void initializeInternal(JNIEnv* env, jobject javaThis, jstring globalConfJson) {
  JNI_METHOD_START
  spotify::jni::JavaString jGlobalConfJson{env, globalConfJson};
  auto dynamic = folly::parseJson(jGlobalConfJson.get());
  auto confArray = ConfigArray::create(dynamic);
  initialize(confArray);
  JNI_METHOD_END()
}

jlong createMemoryManager(JNIEnv* env, jobject javaThis, jobject jListener) {
  JNI_METHOD_START
  auto listener = std::make_unique<BlockAllocationListener>(
      std::make_unique<JavaAllocationListener>(env, jListener), 8 << 10 << 10);
  auto mm = std::make_shared<MemoryManager>(std::move(listener));
  return ObjectStore::global()->save(mm);
  JNI_METHOD_END(-1L)
}

jlong createSession(JNIEnv* env, jobject javaThis, long memoryManagerId) {
  JNI_METHOD_START
  auto mm = ObjectStore::retrieve<MemoryManager>(memoryManagerId);
  return ObjectStore::global()->save(std::make_shared<Session>(mm.get()));
  JNI_METHOD_END(-1L)
}

void releaseCppObject(JNIEnv* env, jobject javaThis, jlong objId) {
  JNI_METHOD_START
  ObjectStore::release(objId);
  JNI_METHOD_END()
}

jint upIteratorAdvance(JNIEnv* env, jobject javaThis, jlong itrId) {
  JNI_METHOD_START
  auto itr = ObjectStore::retrieve<UpIterator>(itrId);
  return static_cast<jint>(itr->advance());
  JNI_METHOD_END(-1)
}

void upIteratorWait(JNIEnv* env, jobject javaThis, jlong itrId) {
  JNI_METHOD_START
  auto itr = ObjectStore::retrieve<UpIterator>(itrId);
  itr->wait();
  JNI_METHOD_END()
}

void blockingQueuePut(
    JNIEnv* env,
    jobject javaThis,
    jlong queueId,
    jlong rowVectorId) {
  JNI_METHOD_START
  auto queue = ObjectStore::retrieve<BlockingQueue>(queueId);
  auto rowVector = ObjectStore::retrieve<RowVector>(rowVectorId);
  queue->put(rowVector);
  JNI_METHOD_END()
}

void blockingQueueNoMoreInput(JNIEnv* env, jobject javaThis, jlong queueId) {
  JNI_METHOD_START
  auto queue = ObjectStore::retrieve<BlockingQueue>(queueId);
  queue->noMoreInput();
  JNI_METHOD_END()
}

void serialTaskAddSplit(
    JNIEnv* env,
    jobject javaThis,
    jlong serialTaskId,
    jstring planNodeId,
    jint groupId,
    jstring connectorSplitJson) {
  JNI_METHOD_START
  auto serialTask = ObjectStore::retrieve<SerialTask>(serialTaskId);
  spotify::jni::JavaString jPlanNodeId{env, planNodeId};
  spotify::jni::JavaString jConnectorSplitJson{env, connectorSplitJson};
  auto jConnectorSplitDynamic = folly::parseJson(jConnectorSplitJson.get());
  auto connectorSplit = std::const_pointer_cast<connector::ConnectorSplit>(
      ISerializable::deserialize<connector::ConnectorSplit>(
          jConnectorSplitDynamic));
  serialTask->addSplit(jPlanNodeId.get(), groupId, connectorSplit);
  JNI_METHOD_END()
}

void serialTaskNoMoreSplits(
    JNIEnv* env,
    jobject javaThis,
    jlong serialTaskId,
    jstring planNodeId) {
  JNI_METHOD_START
  auto serialTask = ObjectStore::retrieve<SerialTask>(serialTaskId);
  spotify::jni::JavaString jPlanNodeId{env, planNodeId};
  serialTask->noMoreSplits(jPlanNodeId.get());
  JNI_METHOD_END()
}

jstring
serialTaskCollectStats(JNIEnv* env, jobject javaThis, jlong serialTaskId) {
  JNI_METHOD_START
  auto serialTask = ObjectStore::retrieve<SerialTask>(serialTaskId);
  const auto stats = serialTask->collectStats();
  const auto statsDynamic = stats->toJson();
  const auto statsJson = folly::toPrettyJson(statsDynamic);
  return env->NewStringUTF(statsJson.data());
  JNI_METHOD_END(nullptr)
}

jstring variantInferType(JNIEnv* env, jobject javaThis, jstring json) {
  JNI_METHOD_START
  spotify::jni::JavaString jJson{env, json};
  auto dynamic = folly::parseJson(jJson.get());
  auto deserialized = variant::create(dynamic);
  auto type = deserialized.inferType();
  auto serializedDynamic = type->serialize();
  auto typeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(typeJson.data());
  JNI_METHOD_END(nullptr);
}

void baseVectorToArrow(
    JNIEnv* env,
    jobject javaThis,
    jlong vectorId,
    jlong cSchema,
    jlong cArray) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  fromBaseVectorToArrow(
      vector,
      reinterpret_cast<struct ArrowSchema*>(cSchema),
      reinterpret_cast<struct ArrowArray*>(cArray));
  JNI_METHOD_END()
}

jstring
baseVectorSerialize(JNIEnv* env, jobject javaThis, jlongArray vectorIds) {
  JNI_METHOD_START
  std::ostringstream out;
  auto safeArray = getLongArrayElementsSafe(env, vectorIds);
  for (int i = 0; i < safeArray.length(); ++i) {
    const jlong& vectorId = safeArray.elems()[i];
    auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
    saveVector(*vector, out);
  }
  auto serializedData = out.str();
  auto encoded =
      encoding::Base64::encode(serializedData.data(), serializedData.size());
  return env->NewStringUTF(encoded.data());
  JNI_METHOD_END(nullptr)
}

jstring baseVectorGetType(JNIEnv* env, jobject javaThis, jlong vectorId) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto serializedDynamic = vector->type()->serialize();
  auto serializeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(serializeJson.data());
  JNI_METHOD_END(nullptr)
}

jint baseVectorGetSize(JNIEnv* env, jobject javaThis, jlong vectorId) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  return static_cast<jint>(vector->size());
  JNI_METHOD_END(-1)
}

jstring baseVectorGetEncoding(JNIEnv* env, jobject javaThis, jlong vectorId) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto name = VectorEncoding::mapSimpleToName(vector->encoding());
  return env->NewStringUTF(name.data());
  JNI_METHOD_END(nullptr)
}

void baseVectorAppend(
    JNIEnv* env,
    jobject javaThis,
    jlong vectorId,
    jlong toAppendVectorId) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto toAppend = ObjectStore::retrieve<BaseVector>(toAppendVectorId);
  vector->append(toAppend.get());
  JNI_METHOD_END()
}

jboolean selectivityVectorIsValid(
    JNIEnv* env,
    jobject javaThis,
    jlong selectivityVectorId,
    jint idx) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<SelectivityVector>(selectivityVectorId);
  auto valid = vector->isValid(static_cast<vector_size_t>(idx));
  return static_cast<jboolean>(valid);
  JNI_METHOD_END(false)
}

jstring iSerializableAsJava(JNIEnv* env, jobject javaThis, jlong id) {
  JNI_METHOD_START
  auto iSerializable = ObjectStore::retrieve<ISerializable>(id);
  auto serializedDynamic = iSerializable->serialize();
  auto serializeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(serializeJson.data());
  JNI_METHOD_END(nullptr)
}

jstring variantAsJava(JNIEnv* env, jobject javaThis, jlong id) {
  JNI_METHOD_START
  auto v = ObjectStore::retrieve<variant>(id);
  auto serializedDynamic = v->serialize();
  auto serializeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(serializeJson.data());
  JNI_METHOD_END(nullptr)
}

jstring tableWriteTraitsOutputType(JNIEnv* env, jobject javaThis) {
  JNI_METHOD_START
  auto type = exec::TableWriteTraits::outputType(nullptr);
  auto serializedDynamic = type->serialize();
  auto typeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(typeJson.data());
  JNI_METHOD_END(nullptr)
}

} // namespace

const char* StaticJniWrapper::getCanonicalName() const {
  return kClassName;
}

void StaticJniWrapper::initialize(JNIEnv* env) {
  JavaClass::setClass(env);

  addNativeMethod(
      "initialize", (void*)initializeInternal, kTypeVoid, kTypeString, nullptr);
  addNativeMethod(
      "createMemoryManager",
      (void*)createMemoryManager,
      kTypeLong,
      "com/facebook/velox4j/memory/AllocationListener",
      nullptr);
  addNativeMethod(
      "createSession", (void*)createSession, kTypeLong, kTypeLong, nullptr);
  addNativeMethod(
      "releaseCppObject",
      (void*)releaseCppObject,
      kTypeVoid,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "upIteratorAdvance",
      (void*)upIteratorAdvance,
      kTypeInt,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "upIteratorWait", (void*)upIteratorWait, kTypeVoid, kTypeLong, nullptr);
  addNativeMethod(
      "blockingQueuePut",
      (void*)blockingQueuePut,
      kTypeVoid,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "blockingQueueNoMoreInput",
      (void*)blockingQueueNoMoreInput,
      kTypeVoid,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "serialTaskAddSplit",
      (void*)serialTaskAddSplit,
      kTypeVoid,
      kTypeLong,
      kTypeString,
      kTypeInt,
      kTypeString,
      nullptr);
  addNativeMethod(
      "serialTaskNoMoreSplits",
      (void*)serialTaskNoMoreSplits,
      kTypeVoid,
      kTypeLong,
      kTypeString,
      nullptr);
  addNativeMethod(
      "serialTaskCollectStats",
      (void*)serialTaskCollectStats,
      kTypeString,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "variantInferType",
      (void*)variantInferType,
      kTypeString,
      kTypeString,
      nullptr);
  addNativeMethod(
      "baseVectorToArrow",
      (void*)baseVectorToArrow,
      kTypeVoid,
      kTypeLong,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "baseVectorSerialize",
      (void*)baseVectorSerialize,
      kTypeString,
      kTypeArray(kTypeLong),
      nullptr);
  addNativeMethod(
      "baseVectorGetType",
      (void*)baseVectorGetType,
      kTypeString,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "baseVectorGetSize",
      (void*)baseVectorGetSize,
      kTypeInt,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "baseVectorGetEncoding",
      (void*)baseVectorGetEncoding,
      kTypeString,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "baseVectorAppend",
      (void*)baseVectorAppend,
      kTypeVoid,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "selectivityVectorIsValid",
      (void*)selectivityVectorIsValid,
      kTypeBool,
      kTypeLong,
      kTypeInt,
      nullptr);
  addNativeMethod(
      "tableWriteTraitsOutputType",
      (void*)tableWriteTraitsOutputType,
      kTypeString,
      nullptr);
  addNativeMethod(
      "iSerializableAsJava",
      (void*)iSerializableAsJava,
      kTypeString,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "variantAsJava", (void*)variantAsJava, kTypeString, kTypeLong, nullptr);

  registerNativeMethods(env);
}

void StaticJniWrapper::mapFields() {
  // No fields to map.
}
} // namespace facebook::velox4j
