# Copyright (c) Meta Platforms, Inc. and affiliates.
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
if(NOT TARGET gflags::gflags)
  find_package(gflags CONFIG QUIET)
endif()

if(NOT TARGET gflags::gflags)
  find_package(gflags QUIET)
endif()

if(NOT TARGET gflags::gflags)
  foreach(
    _nimble_gflags_target
    gflags
    gflags_static
    gflags_nothreads_static
    gflags_shared
    gflags_nothreads_shared
  )
    if(TARGET ${_nimble_gflags_target})
      add_library(nimble_gflags INTERFACE)
      target_link_libraries(nimble_gflags INTERFACE ${_nimble_gflags_target})
      add_library(gflags::gflags ALIAS nimble_gflags)
      break()
    endif()
  endforeach()
endif()

add_executable(nimble_encoding_fuzzer_tests EncodingFuzzerTest.cpp EncodingFuzzer.h)

target_compile_definitions(
  nimble_encoding_fuzzer_tests
  PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
)

# The sources above are compiled with the experimental encodings whatever the
# option says, but nimble_encodings follows the option, and its factories only
# create FrequencyPartition when it is on.
if(NOT NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  target_compile_definitions(
    nimble_encoding_fuzzer_tests
    PRIVATE NIMBLE_ENCODINGS_LIBRARY_WITHOUT_EXPERIMENTAL
  )
endif()

target_link_libraries(
  nimble_encoding_fuzzer_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
  velox_memory
)

add_test(
  nimble_encoding_fuzzer_tests
  nimble_encoding_fuzzer_tests
  --fuzzer_iterations=2
  --fuzzer_max_rows=128
  --nofuzzer_compression
)

add_executable(nimble_encoding_view_fuzzer_tests EncodingViewFuzzerTest.cpp)

target_link_libraries(
  nimble_encoding_view_fuzzer_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
  velox_memory
)

add_test(
  nimble_encoding_view_fuzzer_tests
  nimble_encoding_view_fuzzer_tests
  --view_fuzzer_iterations=2
  --view_fuzzer_max_rows=128
)

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_tuning_fuzzer_tests SubIntSplitTuningFuzzerTest.cpp)
  velox_add_test_headers(nimble_subintsplit_tuning_fuzzer_tests SubIntSplitFuzzer.h)

  target_compile_definitions(
    nimble_subintsplit_tuning_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_tuning_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_tuning_fuzzer_tests
    nimble_subintsplit_tuning_fuzzer_tests
    --sis_tuning_fuzzer_iterations=1
    --sis_tuning_fuzzer_max_rows=256
  )
endif()

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_view_fuzzer_tests SubIntSplitEncodingViewFuzzerTest.cpp)

  target_compile_definitions(
    nimble_subintsplit_view_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_view_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_view_fuzzer_tests
    nimble_subintsplit_view_fuzzer_tests
    --sis_view_fuzzer_iterations=2
    --sis_view_fuzzer_max_rows=128
  )
endif()

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_row_frame_fuzzer_tests SubIntSplitRowFrameFuzzerTest.cpp)

  target_compile_definitions(
    nimble_subintsplit_row_frame_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_row_frame_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_row_frame_fuzzer_tests
    nimble_subintsplit_row_frame_fuzzer_tests
    --sis_row_frame_fuzzer_iterations=1
    --sis_row_frame_fuzzer_max_rows=256
  )
endif()

add_executable(nimble_selection_statistics_fuzzer_tests SelectionStatisticsFuzzerTest.cpp)

target_link_libraries(
  nimble_selection_statistics_fuzzer_tests
  nimble_encodings
  nimble_common
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
)

add_test(
  nimble_selection_statistics_fuzzer_tests
  nimble_selection_statistics_fuzzer_tests
  --selection_fuzzer_iterations=2
)

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_key_reorder_fuzzer_tests SubIntSplitKeyReorderFuzzerTest.cpp)

  target_compile_definitions(
    nimble_subintsplit_key_reorder_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_key_reorder_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_key_reorder_fuzzer_tests
    nimble_subintsplit_key_reorder_fuzzer_tests
    --sis_reorder_fuzzer_iterations=1
    --sis_reorder_fuzzer_max_rows=256
  )
endif()

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_planner_fuzzer_tests SubIntSplitPlannerFuzzerTest.cpp)

  target_compile_definitions(
    nimble_subintsplit_planner_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_planner_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_planner_fuzzer_tests
    nimble_subintsplit_planner_fuzzer_tests
    --sis_planner_fuzzer_iterations=1
    --sis_planner_fuzzer_max_rows=256
  )
endif()

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_executable(nimble_subintsplit_admission_fuzzer_tests SubIntSplitAdmissionFuzzerTest.cpp)

  target_compile_definitions(
    nimble_subintsplit_admission_fuzzer_tests
    PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  )

  target_link_libraries(
    nimble_subintsplit_admission_fuzzer_tests
    nimble_encodings_tests_utils
    nimble_encodings
    nimble_common
    gtest
    gflags::gflags
    glog::glog
    Folly::folly
    velox_memory
  )

  add_test(
    nimble_subintsplit_admission_fuzzer_tests
    nimble_subintsplit_admission_fuzzer_tests
    --sis_admission_fuzzer_iterations=1
    --sis_admission_fuzzer_max_rows=512
  )
endif()

add_executable(nimble_estimator_refinement_fuzzer_tests EstimatorRefinementFuzzerTest.cpp)

target_link_libraries(
  nimble_estimator_refinement_fuzzer_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_common
  gtest
  gflags::gflags
  glog::glog
  Folly::folly
  velox_memory
)

add_test(
  nimble_estimator_refinement_fuzzer_tests
  nimble_estimator_refinement_fuzzer_tests
  --estimator_fuzzer_iterations=2
  --estimator_fuzzer_max_rows=1024
)
