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
#include <chrono>
#include <thread>

namespace facebook::velox::parquet {

template <typename Func>
void retry(int attempts, std::chrono::milliseconds delay, Func func) {
  for (int i = 0; i < attempts; ++i) {
    try {
      func();
      return; // Success
    } catch (const std::exception& e) {
      if (i < attempts - 1) {
        std::this_thread::sleep_for(delay); // Wait before retrying
      } else {
        throw e;
      }
    }
  }
}

} // namespace facebook::velox::parquet
