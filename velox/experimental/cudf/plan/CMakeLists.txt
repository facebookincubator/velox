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

add_library(velox_cudf_plan CudfPlanNodeChecker.cpp CudfExpressionChecker.cpp)

target_link_libraries(
  velox_cudf_plan
  PRIVATE
    velox_cudf_hive_connector
    velox_core
    velox_exception
    velox_expression
    velox_cudf_expression_registry
    velox_cudf_exec
)
