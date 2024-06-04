# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(BENCHMARK_DEPENDENCIES_NO_FUNC
    fmt::fmt
    velox_functions_test_lib
    velox_exec
    velox_exec_test_lib
    velox_vector_test_lib
    gflags::gflags
    Folly::folly
    ${FOLLY_BENCHMARK})

set(BENCHMARK_DEPENDENCIES
    velox_window
    velox_aggregates
    velox_hive_connector
    velox_functions_prestosql
    velox_functions_lib
    velox_vector_fuzzer
    ${BENCHMARK_DEPENDENCIES_NO_FUNC})

add_executable(velox_windows_benchmarks WindowBenchmark.cpp)

target_link_libraries(velox_windows_benchmarks ${BENCHMARK_DEPENDENCIES})
