/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Standalone runner for the Nimble writer fuzzer, used for manual soaks and
// packaged for Cogwheel. Unlike the gtest binary it runs only the fuzzer loop,
// and shares its setup and run logic via NimbleWriterFuzzerRunner.

#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzerRunner.h"

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::fuzzer::setUpFuzzerEnvironments();
  facebook::nimble::fuzzer::runNimbleWriterFuzzer();
  return 0;
}
