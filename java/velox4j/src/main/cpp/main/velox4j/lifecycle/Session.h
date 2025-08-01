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

#include <memory>

#include "velox4j/lifecycle/ObjectStore.h"
#include "velox4j/memory/MemoryManager.h"

namespace facebook::velox4j {

// A JNI session that is bound to a JniWrapper.
class Session {
 public:
  explicit Session(MemoryManager* memoryManager)
      : memoryManager_(memoryManager), objectStore_(ObjectStore::create()) {}

  virtual ~Session() = default;

  MemoryManager* memoryManager() {
    return memoryManager_;
  }

  ObjectStore* objectStore() {
    return objectStore_.get();
  }

 private:
  MemoryManager* memoryManager_;
  std::unique_ptr<ObjectStore> objectStore_;
};

} // namespace facebook::velox4j
