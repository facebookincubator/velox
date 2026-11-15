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

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TORCHWAVE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Generate element_test.pt2 and element_test_results.pt at build time.
add_custom_command(
  OUTPUT
    ${TORCHWAVE_TEST_DATA_DIR}/element_test.pt2
    ${TORCHWAVE_TEST_DATA_DIR}/element_test_results.pt
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/element_test_run_pt2.py --output_dir
    ${TORCHWAVE_TEST_DATA_DIR}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/element_test_run_pt2.py
    ${CMAKE_CURRENT_SOURCE_DIR}/element_test.py
  COMMENT "Generating element_test.pt2 and reference results"
)

add_custom_target(
  element_test_data
  DEPENDS
    ${TORCHWAVE_TEST_DATA_DIR}/element_test.pt2
    ${TORCHWAVE_TEST_DATA_DIR}/element_test_results.pt
)

add_library(torchwave_executor_test_base CompiledPlan.cpp DataGen.cpp ExecutorTestBase.cpp)

target_link_libraries(
  torchwave_executor_test_base
  torch_wave
  torch_wave_describe_pt
  GTest::gtest
  Folly::folly
  CUDA::cudart
)

add_executable(torchwave_executor_test ExecutorTest.cpp)

add_dependencies(torchwave_executor_test element_test_data)

target_link_libraries(
  torchwave_executor_test
  torchwave_executor_test_base
  torch_wave
  GTest::gtest
  Folly::folly
  CUDA::cudart
)

add_test(
  NAME torchwave_executor_test
  COMMAND torchwave_executor_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(torchwave_executor_test PROPERTIES LABELS cuda_driver ENVIRONMENT FB_TEST=1)

add_executable(torchwave_alloc_group_test AllocGroupTest.cpp)

target_link_libraries(torchwave_alloc_group_test torch_wave GTest::gtest Folly::folly CUDA::cudart)

add_test(
  NAME torchwave_alloc_group_test
  COMMAND torchwave_alloc_group_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(torchwave_alloc_group_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_launch_partition_test LaunchPartitionTest.cpp)

target_link_libraries(torchwave_launch_partition_test torch_wave GTest::gtest GTest::gtest_main)

add_test(NAME torchwave_launch_partition_test COMMAND torchwave_launch_partition_test)

add_executable(torchwave_model_test ModelTest.cpp)

target_link_libraries(
  torchwave_model_test
  torchwave_executor_test_base
  torch_wave
  GTest::gtest
  GTest::gtest_main
  CUDA::cudart
)

add_test(
  NAME torchwave_model_test
  COMMAND torchwave_model_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(torchwave_model_test PROPERTIES LABELS cuda_driver ENVIRONMENT FB_TEST=1)

add_executable(torchwave_compile_test CompileTest.cpp)

target_link_libraries(
  torchwave_compile_test
  torchwave_executor_test_base
  torch_wave
  re2::re2
  GTest::gtest
  Folly::folly
  CUDA::cudart
)

add_test(
  NAME torchwave_compile_test
  COMMAND torchwave_compile_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(torchwave_compile_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_kernel_test KernelTest.cu)

target_compile_options(torchwave_kernel_test PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(
  torchwave_kernel_test
  torch_wave
  fmt::fmt
  gflags::gflags
  GTest::gtest
  Folly::folly
  CUDA::cudart
)

add_test(NAME torchwave_kernel_test COMMAND torchwave_kernel_test)
set_tests_properties(torchwave_kernel_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_kernel_ld_test KernelLdTest.cu)

target_compile_options(torchwave_kernel_ld_test PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(
  torchwave_kernel_ld_test
  torch
  GTest::gtest
  GTest::gtest_main
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::cudart
)

target_include_directories(torchwave_kernel_ld_test SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})

add_test(NAME torchwave_kernel_ld_test COMMAND torchwave_kernel_ld_test)
set_tests_properties(torchwave_kernel_ld_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_grid_benchmark GridBenchmark.cu)

target_compile_options(torchwave_grid_benchmark PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(
  torchwave_grid_benchmark
  torch_wave
  gflags::gflags
  GTest::gtest
  GTest::gtest_main
  CUDA::cudart
)

add_test(NAME torchwave_grid_benchmark COMMAND torchwave_grid_benchmark)
set_tests_properties(torchwave_grid_benchmark PROPERTIES LABELS cuda_driver)

add_executable(torchwave_tensor_test TensorTest.cu)

target_compile_options(torchwave_tensor_test PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(torchwave_tensor_test torch_wave GTest::gtest GTest::gtest_main CUDA::cudart)

add_test(NAME torchwave_tensor_test COMMAND torchwave_tensor_test)
set_tests_properties(torchwave_tensor_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_cuda_benchmarks CudaBenchmarks.cu)

target_compile_options(torchwave_cuda_benchmarks PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(torchwave_cuda_benchmarks torch_wave gflags::gflags CUDA::cudart)

add_executable(torchwave_scatter_add_benchmark scatterAddBenchmark.cu)

target_compile_options(torchwave_scatter_add_benchmark PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(torchwave_scatter_add_benchmark torch_wave gflags::gflags CUDA::cudart)

add_executable(torchwave_reverse_benchmark reverseBenchmark.cu)

target_compile_options(torchwave_reverse_benchmark PRIVATE ${VELOX_CUDA_DEBUG_FLAGS})

target_link_libraries(torchwave_reverse_benchmark gflags::gflags CUDA::cudart)

add_executable(torchwave_graph_tool GraphTool.cpp)

target_link_libraries(
  torchwave_graph_tool
  torch_wave
  torch_wave_describe_pt
  torch_wave_graph_view
  gflags::gflags
  Folly::folly
)
