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

#include <folly/init/Init.h>

#include "velox/common/fuzzer/Utils.h"
#include "velox/expression/fuzzer/ExpressionFuzzerUtils.h"
#include "velox/expression/fuzzer/FuzzerRunner.h"

using namespace facebook::velox;
using facebook::velox::exec::test::PrestoQueryRunner;
using facebook::velox::fuzzer::ExpressionFuzzer;
using facebook::velox::fuzzer::FuzzerRunner;
using facebook::velox::test::ReferenceQueryRunner;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});

  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};

  auto referenceQueryRunner = initializeExpressionFuzzer(rootPool);

  size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  // To emphasize fuzzing the special forms and their optimizations, remove
  // a good chunk of functions to choose from by coin toss at 80% rate.
  // The selected functions are added to the skipFunctions set.
  // The actual list of function may be further reduced by the "only" function
  // list.
  folly::Random::DefaultGenerator rng(initialSeed);
  auto signatures = facebook::velox::getFunctionSignatures();
  for (const auto& signature : signatures) {
    skipFunctions.insert(signature.first);
  }
  LOG(INFO) << "Total number of skipped functions: " << skipFunctions.size();

  FuzzerRunner::runFromGtest(
      initialSeed,
      skipFunctions,
      exprTransformers,
      {{"session_timezone", "America/Los_Angeles"},
       {"adjust_timestamp_to_session_timezone", "true"}},
      argTypesGenerators,
      argValuesGenerators,
      referenceQueryRunner.value_or(nullptr),
      std::make_shared<
          facebook::velox::fuzzer::SpecialFormSignatureGenerator>());
}
