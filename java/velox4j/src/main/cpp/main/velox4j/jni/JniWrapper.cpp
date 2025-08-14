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
#include "velox4j/jni/JniWrapper.h"

#include <ClassRegistry.h>
#include <JavaString.h>
#include <JniTypes.h>
#include <fmt/core.h>
#include <folly/json/json.h>
#include <jni_md.h>
#include <velox/common/base/Exceptions.h>
#include <velox/common/encode/Base64.h>
#include <velox/common/future/VeloxPromise.h>
#include <velox/common/memory/MemoryPool.h>
#include <velox/common/serialization/Serializable.h>
#include <velox/core/PlanNode.h>
#include <velox/exec/TableWriter.h>
#include <velox/type/Type.h>
#include <velox/type/Variant.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/ComplexVector.h>
#include <velox/vector/SelectivityVector.h>
#include <velox/vector/TypeAliases.h>
#include <velox/vector/VectorSaver.h>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "velox4j/arrow/Arrow.h"
#include "velox4j/connector/ExternalStream.h"
#include "velox4j/eval/Evaluator.h"
#include "velox4j/iterator/BlockingQueue.h"
#include "velox4j/iterator/DownIterator.h"
#include "velox4j/iterator/UpIterator.h"
#include "velox4j/jni/JniCommon.h"
#include "velox4j/jni/JniError.h"
#include "velox4j/lifecycle/ObjectStore.h"
#include "velox4j/lifecycle/Session.h"
#include "velox4j/memory/MemoryManager.h"
#include "velox4j/query/Query.h"
#include "velox4j/query/QueryExecutor.h"

namespace facebook::velox4j {
using namespace facebook::velox;

namespace {
const char* kClassName = "com/facebook/velox4j/jni/JniWrapper";

// Get the Velox4J session object that is associated with the current
// JniWrapper.
Session* sessionOf(JNIEnv* env, jobject javaThis) {
  static const auto* clazz = jniClassRegistry()->get(kClassName);
  static jmethodID methodId = clazz->getMethod("sessionId");
  const jlong sessionId = env->CallLongMethod(javaThis, methodId);
  checkException(env);
  return ObjectStore::retrieve<Session>(sessionId).get();
}

jlong createEvaluator(JNIEnv* env, jobject javaThis, jstring evalJson) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  spotify::jni::JavaString jExprJson{env, evalJson};

  auto evaluationSerdePool = session->memoryManager()->getVeloxPool(
      "Evaluation Serde Memory Pool", memory::MemoryPool::Kind::kLeaf);
  auto exprDynamic = folly::parseJson(jExprJson.get());
  auto evaluation =
      ISerializable::deserialize<Evaluation>(exprDynamic, evaluationSerdePool);

  auto evaluator =
      std::make_shared<Evaluator>(session->memoryManager(), evaluation);
  return sessionOf(env, javaThis)->objectStore()->save(evaluator);
  JNI_METHOD_END(-1L)
}

jlong evaluatorEval(
    JNIEnv* env,
    jobject javaThis,
    jlong evaluatorId,
    jlong selectivityVectorId,
    jlong rowVectorId) {
  JNI_METHOD_START
  auto evaluator = ObjectStore::retrieve<Evaluator>(evaluatorId);
  auto selectivityVector =
      ObjectStore::retrieve<SelectivityVector>(selectivityVectorId);
  auto input = ObjectStore::retrieve<RowVector>(rowVectorId);
  return sessionOf(env, javaThis)
      ->objectStore()
      ->save(evaluator->eval(*selectivityVector, *input));
  JNI_METHOD_END(-1L)
}

jlong createQueryExecutor(JNIEnv* env, jobject javaThis, jstring queryJson) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  spotify::jni::JavaString jQueryJson{env, queryJson};
  auto querySerdePool = session->memoryManager()->getVeloxPool(
      fmt::format("Query Serde Memory Pool"), memory::MemoryPool::Kind::kLeaf);
  // Keep the pool alive until the task is finished.
  auto queryDynamic = folly::parseJson(jQueryJson.get());
  auto query = ISerializable::deserialize<Query>(queryDynamic, querySerdePool);
  auto exec = std::make_shared<QueryExecutor>(session->memoryManager(), query);
  return sessionOf(env, javaThis)->objectStore()->save(exec);
  JNI_METHOD_END(-1L)
}

jlong queryExecutorExecute(
    JNIEnv* env,
    jobject javaThis,
    jlong queryExecutorId) {
  JNI_METHOD_START
  auto exec = ObjectStore::retrieve<QueryExecutor>(queryExecutorId);
  return sessionOf(env, javaThis)
      ->objectStore()
      ->save<SerialTask>(exec->execute());
  JNI_METHOD_END(-1L)
}

jlong upIteratorGet(JNIEnv* env, jobject javaThis, jlong itrId) {
  JNI_METHOD_START
  auto itr = ObjectStore::retrieve<UpIterator>(itrId);
  return sessionOf(env, javaThis)->objectStore()->save(itr->get());
  JNI_METHOD_END(-1L)
}

jlong createExternalStreamFromDownIterator(
    JNIEnv* env,
    jobject javaThis,
    jobject itrRef) {
  JNI_METHOD_START
  auto es = std::make_shared<DownIterator>(env, itrRef);
  return sessionOf(env, javaThis)->objectStore()->save(es);
  JNI_METHOD_END(-1L)
}

jlong createBlockingQueue(JNIEnv* env, jobject javaThis) {
  JNI_METHOD_START
  auto queue = std::make_shared<BlockingQueue>();
  return sessionOf(env, javaThis)->objectStore()->save(queue);
  JNI_METHOD_END(-1L)
}

jlong createEmptyBaseVector(JNIEnv* env, jobject javaThis, jstring typeJson) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  auto serdePool = session->memoryManager()->getVeloxPool(
      "Serde Memory Pool", memory::MemoryPool::Kind::kLeaf);
  spotify::jni::JavaString jTypeJson{env, typeJson};
  auto dynamic = folly::parseJson(jTypeJson.get());
  auto type = ISerializable::deserialize<Type>(dynamic, serdePool);
  auto vectorPool = session->memoryManager()->getVeloxPool(
      "BaseVector Memory Pool", memory::MemoryPool::Kind::kLeaf);
  auto vector = BaseVector::create(type, 0, vectorPool);
  return session->objectStore()->save(vector);
  JNI_METHOD_END(-1L)
}

jlong arrowToBaseVector(
    JNIEnv* env,
    jobject javaThis,
    jlong cSchema,
    jlong cArray) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  auto pool = session->memoryManager()->getVeloxPool(
      "Arrow Import Memory Pool", memory::MemoryPool::Kind::kLeaf);
  auto vector = fromArrowToBaseVector(
      pool,
      reinterpret_cast<struct ArrowSchema*>(cSchema),
      reinterpret_cast<struct ArrowArray*>(cArray));
  return session->objectStore()->save(vector);
  JNI_METHOD_END(-1L)
}

jlongArray
baseVectorDeserialize(JNIEnv* env, jobject javaThis, jstring serialized) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  spotify::jni::JavaString jSerialized{env, serialized};
  auto decoded = encoding::Base64::decode(jSerialized.get());
  std::istringstream dataStream(decoded);
  auto pool = session->memoryManager()->getVeloxPool(
      "Decoding Memory Pool", memory::MemoryPool::Kind::kLeaf);
  std::vector<ObjectHandle> vectorIds{};
  while (dataStream.tellg() < decoded.size()) {
    const VectorPtr& vector = restoreVector(dataStream, pool);
    const ObjectHandle vectorId = session->objectStore()->save(vector);
    vectorIds.push_back(vectorId);
  }
  const jsize& len = static_cast<jsize>(vectorIds.size());
  const jlongArray& out = env->NewLongArray(len);
  env->SetLongArrayRegion(out, 0, len, vectorIds.data());
  return out;
  JNI_METHOD_END(nullptr)
}

jlong baseVectorWrapInConstant(
    JNIEnv* env,
    jobject javaThis,
    jlong vectorId,
    jint length,
    jint index) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto constVector = BaseVector::wrapInConstant(length, index, vector);
  return sessionOf(env, javaThis)->objectStore()->save(constVector);
  JNI_METHOD_END(-1)
}

jlong baseVectorSlice(
    JNIEnv* env,
    jobject javaThis,
    jlong vectorId,
    jint offset,
    jint length) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto slicedVector = vector->slice(offset, length);
  return sessionOf(env, javaThis)->objectStore()->save(slicedVector);
  JNI_METHOD_END(-1)
}

jlong baseVectorLoadedVector(JNIEnv* env, jobject javaThis, jlong vectorId) {
  JNI_METHOD_START
  auto vector = ObjectStore::retrieve<BaseVector>(vectorId);
  auto loadedVector = BaseVector::loadedVectorShared(vector);
  return sessionOf(env, javaThis)->objectStore()->save(loadedVector);
  JNI_METHOD_END(-1)
}

jlong createSelectivityVector(JNIEnv* env, jobject javaThis, jint length) {
  JNI_METHOD_START
  auto vector =
      std::make_shared<SelectivityVector>(static_cast<vector_size_t>(length));
  return sessionOf(env, javaThis)->objectStore()->save(vector);
  JNI_METHOD_END(-1)
}

jstring tableWriteTraitsOutputTypeWithAggregationNode(
    JNIEnv* env,
    jobject javaThis,
    jstring aggregationNodeJson) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  spotify::jni::JavaString jJson{env, aggregationNodeJson};
  auto dynamic = folly::parseJson(jJson.get());
  auto serdePool = session->memoryManager()->getVeloxPool(
      "Serde Memory Pool", memory::MemoryPool::Kind::kLeaf);
  auto aggregationNode = std::const_pointer_cast<core::AggregationNode>(
      ISerializable::deserialize<core::AggregationNode>(dynamic, serdePool));
  auto type = exec::TableWriteTraits::outputType(aggregationNode);
  auto serializedDynamic = type->serialize();
  auto typeJson = folly::toPrettyJson(serializedDynamic);
  return env->NewStringUTF(typeJson.data());
  JNI_METHOD_END(nullptr)
}

jlong iSerializableAsCpp(JNIEnv* env, jobject javaThis, jstring json) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  auto serdePool = session->memoryManager()->getVeloxPool(
      "Serde Memory Pool", memory::MemoryPool::Kind::kLeaf);
  spotify::jni::JavaString jJson{env, json};
  auto dynamic = folly::parseJson(jJson.get());
  auto deserialized = std::const_pointer_cast<ISerializable>(
      ISerializable::deserialize<ISerializable>(dynamic, serdePool));
  return session->objectStore()->save(deserialized);
  JNI_METHOD_END(-1)
}

jlong variantAsCpp(JNIEnv* env, jobject javaThis, jstring json) {
  JNI_METHOD_START
  auto session = sessionOf(env, javaThis);
  spotify::jni::JavaString jJson{env, json};
  auto dynamic = folly::parseJson(jJson.get());
  auto deserialized = variant::create(dynamic);
  return session->objectStore()->save(std::make_shared<variant>(deserialized));
  JNI_METHOD_END(-1)
}

class ExternalStreamAsUpIterator : public UpIterator {
 public:
  explicit ExternalStreamAsUpIterator(const std::shared_ptr<ExternalStream>& es)
      : es_(es) {}

  State advance() override {
    VELOX_CHECK_NULL(pending_);
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto out = es_->read(future);
    if (out == std::nullopt) {
      VELOX_CHECK(future.valid());
      // Do not wait for the future to be fulfilled, just return.
      return State::BLOCKED;
    }
    VELOX_CHECK(!future.valid());
    if (out == nullptr) {
      return State::FINISHED;
    }
    pending_ = out.value();
    return State::AVAILABLE;
  }

  void wait() override {
    VELOX_CHECK_NULL(pending_);
    VELOX_NYI("Not implemented: {}", __func__);
  }

  RowVectorPtr get() override {
    VELOX_CHECK_NOT_NULL(
        pending_,
        "ExternalStreamAsUpIterator: No pending row vector to return. Make "
        "sure the iterator is available via member function advance() first");
    auto out = pending_;
    pending_ = nullptr;
    return out;
  }

 private:
  const std::shared_ptr<ExternalStream> es_;
  RowVectorPtr pending_{nullptr};
};

jlong createUpIteratorWithExternalStream(
    JNIEnv* env,
    jobject javaThis,
    jlong id) {
  JNI_METHOD_START
  auto es = ObjectStore::retrieve<ExternalStream>(id);
  return sessionOf(env, javaThis)
      ->objectStore()
      ->save(std::make_shared<ExternalStreamAsUpIterator>(es));
  JNI_METHOD_END(-1L)
}
} // namespace

void JniWrapper::mapFields() {
  // No fields to map.
}

const char* JniWrapper::getCanonicalName() const {
  return kClassName;
}

void JniWrapper::initialize(JNIEnv* env) {
  JavaClass::setClass(env);

  cacheMethod(env, "sessionId", kTypeLong, nullptr);
  addNativeMethod(
      "createEvaluator",
      (void*)createEvaluator,
      kTypeLong,
      kTypeString,
      nullptr);
  addNativeMethod(
      "evaluatorEval",
      (void*)evaluatorEval,
      kTypeLong,
      kTypeLong,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "createQueryExecutor",
      (void*)createQueryExecutor,
      kTypeLong,
      kTypeString,
      nullptr);
  addNativeMethod(
      "queryExecutorExecute",
      (void*)queryExecutorExecute,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "upIteratorGet", (void*)upIteratorGet, kTypeLong, kTypeLong, nullptr);
  addNativeMethod(
      "createExternalStreamFromDownIterator",
      (void*)createExternalStreamFromDownIterator,
      kTypeLong,
      "com/facebook/velox4j/iterator/DownIterator",
      nullptr);
  addNativeMethod(
      "createBlockingQueue", (void*)createBlockingQueue, kTypeLong, nullptr);
  addNativeMethod(
      "createEmptyBaseVector",
      (void*)createEmptyBaseVector,
      kTypeLong,
      kTypeString,
      nullptr);
  addNativeMethod(
      "arrowToBaseVector",
      (void*)arrowToBaseVector,
      kTypeLong,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "baseVectorDeserialize",
      (void*)baseVectorDeserialize,
      kTypeArray(kTypeLong),
      kTypeString,
      nullptr);
  addNativeMethod(
      "baseVectorWrapInConstant",
      (void*)baseVectorWrapInConstant,
      kTypeLong,
      kTypeLong,
      kTypeInt,
      kTypeInt,
      nullptr);
  addNativeMethod(
      "baseVectorSlice",
      (void*)baseVectorSlice,
      kTypeLong,
      kTypeLong,
      kTypeInt,
      kTypeInt,
      nullptr);
  addNativeMethod(
      "baseVectorLoadedVector",
      (void*)baseVectorLoadedVector,
      kTypeLong,
      kTypeLong,
      nullptr);
  addNativeMethod(
      "createSelectivityVector",
      (void*)createSelectivityVector,
      kTypeLong,
      kTypeInt,
      nullptr);
  addNativeMethod(
      "tableWriteTraitsOutputTypeWithAggregationNode",
      (void*)tableWriteTraitsOutputTypeWithAggregationNode,
      kTypeString,
      kTypeString,
      nullptr);
  addNativeMethod(
      "iSerializableAsCpp",
      (void*)iSerializableAsCpp,
      kTypeLong,
      kTypeString,
      nullptr);
  addNativeMethod(
      "variantAsCpp", (void*)variantAsCpp, kTypeLong, kTypeString, nullptr);
  addNativeMethod(
      "createUpIteratorWithExternalStream",
      (void*)createUpIteratorWithExternalStream,
      kTypeLong,
      kTypeLong,
      nullptr);

  registerNativeMethods(env);
}

} // namespace facebook::velox4j
