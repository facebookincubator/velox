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

add_library(
  velox_cudf_exec
  AggregationRegistry.cpp
  CudfAggregation.cpp
  CudfAssignUniqueId.cpp
  CudfBatchConcat.cpp
  CudfConversion.cpp
  CudfDistinct.cpp
  CudfEnforceSingleRow.cpp
  CudfFilterProject.cpp
  CudfGroupId.cpp
  CudfGroupby.cpp
  CudfHashJoin.cpp
  CudfLimit.cpp
  CudfLocalPartition.cpp
  CudfMarkDistinct.cpp
  CudfNestedLoopJoin.cpp
  CudfOrderBy.cpp
  CudfReduce.cpp
  CudfTopN.cpp
  CudfTopNRowNumber.cpp
  CudfWindow.cpp
  DebugUtil.cpp
  DecimalAggregationDevice.cu
  DecimalAggregationHostOps.cpp
  DecimalAggregationState.cpp
  GpuResources.cpp
  OperatorAdapters.cpp
  PrestoAggregateFunctions.cpp
  SparkAggregateFunctions.cpp
  ToCudf.cpp
  Utilities.cpp
  VeloxCudfInterop.cpp
)

target_link_libraries(
  velox_cudf_exec
  PUBLIC cudf::cudf nvtx3::nvtx3-cpp
  PRIVATE
    arrow
    velox_arrow_bridge
    velox_common_base
    velox_cudf_expression
    velox_cudf_hive_connector
    velox_cudf_iceberg_connector
    velox_cudf_vector
    velox_exception
    velox_exec
    velox_test_util
)

# The ucx-exchange library is only built when VELOX_ENABLE_UCX_EXCHANGE is on,
# so the calls into it -- registerUcxTransports() in ToCudf.cpp and the UCX
# operator adapters in OperatorAdapters.cpp -- are compiled out otherwise.
# Without this, configuring with VELOX_ENABLE_CUDF=ON and the UCX exchange off
# fails: the target does not exist and ucxx was never fetched, so the
# ucx-exchange headers do not resolve either.
if(VELOX_ENABLE_UCX_EXCHANGE)
  target_link_libraries(velox_cudf_exec PRIVATE velox_ucx_exchange)
  target_compile_definitions(velox_cudf_exec PRIVATE VELOX_ENABLE_UCX_EXCHANGE)
endif()

target_compile_options(velox_cudf_exec PRIVATE -Wno-missing-field-initializers)
