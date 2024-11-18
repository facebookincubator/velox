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
#include "velox/jni/cpp/expression/ExpressionUtils.hpp"
#include "velox/jni/cpp/jni/JniErrors.h"
#include "velox/jni/cpp/jni/JniUtil.h"

#include <iostream>

#include "MemoryManager.h"


#include "velox/jni/cpp/memory/NativeMemoryManager.hpp"

namespace facebook::velox::sdk::memory {

using namespace facebook::velox::sdk::jni;

std::string NativeMemoryManager::CLASS_NAME =
    "org/apache/spark/rpc/jni/NativeRPCManager";

void NativeMemoryManager::initInternal() {
  addNativeMethod(
      "nativeCreate", reinterpret_cast<void*>(nativeCreate), kTypeLong, NULL);
  addNativeMethod(
    "nativeMemoryStatics", reinterpret_cast<void*>(nativeMemoryStatics), kTypeString, NULL);
}

jlong NativeMemoryManager::nativeCreate(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
  auto* handle = new SharedPtrHandle{MemoryManager::get()};
  return reinterpret_cast<long>(handle);
  JNI_METHOD_END(-1)
}
jstring NativeMemoryManager::nativeMemoryStatics(JNIEnv* env, jobject obj) {
  JNI_METHOD_START
   std::shared_ptr<MemoryManager> memoryManager =  as<MemoryManager>(obj);
  return env->NewStringUTF(memoryManager->memoryStatics().c_str());
  JNI_METHOD_END(env->NewStringUTF(""))
}

} // namespace facebook::velox::sdk