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

#include <jni.h>
#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/PartitionFunction.h"

#include <glog/logging.h>
#include <serializers/PrestoSerializer.h>
#include <velox/jni/cpp/memory/NativeMemoryManager.hpp>

#include "JniCommon.h"
#include "velox/jni/cpp/functions/RegistrationAllFunctions.h"
#include "velox/jni/cpp/native/NativePlanBuilder.hpp"

#include "velox/jni/cpp/jni/JniErrors.h"

#include "JniUtil.h"

namespace facebook::velox::sdk::memory {}
namespace facebook::velox::sdk::execution {
class NativeColumnarExecution;
}
using namespace facebook;

#ifdef __cplusplus
extern "C" {
#endif

using namespace velox::sdk;
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), jniVersion) != JNI_OK) {
    return JNI_ERR;
  }
  // logging
  velox::sdk::registerAllFunctions();
  google::InitGoogleLogging("velox");
  FLAGS_logtostderr = true;
  getJniErrorsState()->initialize(env);
  JniUtil::Init(vm);
  velox::Type::registerSerDe();
  velox::core::ITypedExpr::registerSerDe();
  velox::core::PlanNode::registerSerDe();
  FLAGS_experimental_enable_legacy_cast = false;
  jni::NativeClass::RegisterNatives<NativePlanBuilder>();
  velox::serializer::presto::PrestoVectorSerde::registerVectorSerde();
  velox::exec::registerPartitionFunctionSerDe();
  std::make_shared<memory::NativeMemoryManager>();
  return jniVersion;
}

void JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  vm->GetEnv(reinterpret_cast<void**>(&env), jniVersion);
  google::ShutdownGoogleLogging();
}

#ifdef __cplusplus
}
#endif
