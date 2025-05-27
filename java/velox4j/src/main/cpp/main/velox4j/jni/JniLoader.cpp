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
#include <JniHelpers.h>
#include <glog/logging.h>
#include <jni.h>
#include "JniWrapper.h"
#include "StaticJniWrapper.h"
#include "velox4j/iterator/DownIterator.h"
#include "velox4j/jni/JniCommon.h"
#include "velox4j/jni/JniError.h"
#include "velox4j/memory/JavaAllocationListener.h"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* jvm, void*) {
  LOG(INFO) << "Initializing Velox4J...";
  JNIEnv* env = jniHelpersInitialize(jvm);
  if (env == nullptr) {
    return -1;
  }

  velox4j::getJniErrorState()->ensureInitialized(env);
  velox4j::jniClassRegistry()->add(env, new velox4j::StaticJniWrapper(env));
  velox4j::jniClassRegistry()->add(env, new velox4j::JniWrapper(env));
  velox4j::jniClassRegistry()->add(
      env, new velox4j::DownIteratorJniWrapper(env));
  velox4j::jniClassRegistry()->add(
      env, new velox4j::JavaAllocationListenerJniWrapper(env));

  LOG(INFO) << "Velox4J initialized.";
  return JAVA_VERSION;
}
