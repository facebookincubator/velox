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

#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

namespace facebook::velox::exec::test {

/// Common entry point for the aggregate fuzzer.
/// Takes the command line arguments along with functions to be skipped,
/// and the query runner.
int runAggregationFuzzer(
    int argc,
    char** argv,
    const std::unordered_set<std::string>& skipFunctions,
    std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner);

} // namespace facebook::velox::exec::test
