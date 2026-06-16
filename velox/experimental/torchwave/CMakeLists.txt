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

find_package(nlohmann_json QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(
  torch_wave
  Builtins.cpp
  Cat.cpp
  Compile.cpp
  CompiledOp.cpp
  Elementwise.cpp
  Executor.cpp
  GraphOptimizer.cpp
  KernelOperation.cpp
  NodePrinter.cpp
  ParallelExpr.cpp
  Project.cpp
  Pt2Load.cpp
  Registry.cpp
  Utils.cpp
  WaveGraph.cpp
)

target_link_libraries(
  torch_wave
  velox_wave_common
  torch
  torch_cpu
  c10
  fmt::fmt
  nlohmann_json::nlohmann_json
  Folly::folly
  CUDA::cudart
)

target_include_directories(torch_wave SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

add_library(torch_wave_describe_pt DescribePt.cpp)

target_link_libraries(torch_wave_describe_pt torch torch_cpu c10)

target_include_directories(torch_wave_describe_pt SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

add_library(torch_wave_graph_view GraphView.cpp)

target_link_libraries(torch_wave_graph_view torch_wave)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
