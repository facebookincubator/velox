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

add_subdirectory(specialforms)
velox_add_library(
  velox_functions_spark_impl
  ArrayGetFunction.cpp
  ArraySort.cpp
  AssertNotNull.cpp
  SequenceFunction.cpp
  Transform.cpp
  CharVarcharUtils.cpp
  Comparisons.cpp
  ConcatWs.cpp
  DecimalArithmetic.cpp
  DecimalCeil.cpp
  DecimalCompare.cpp
  FilterFunction.cpp
  Hash.cpp
  In.cpp
  LeastGreatest.cpp
  MakeTimestamp.cpp
  Map.cpp
  RegexFunctions.cpp
  Size.cpp
  String.cpp
  TimestampUtils.cpp
  Transform.cpp
  UnscaledValueFunction.cpp
  HEADERS
  Arena.h
  Arithmetic.h
  ArrayAppend.h
  ArrayConcat.h
  ArrayFlattenFunction.h
  ArrayInsert.h
  ArrayMinMaxFunction.h
  ArrayPrepend.h
  ArraySort.h
  AssertNotNull.h
  Base64Function.h
  Bitwise.h
  CharTypeWriteSideCheck.h
  CharVarcharUtils.h
  Comparisons.h
  ConcatWs.h
  DateTimeFunctions.h
  DecimalArithmetic.h
  DecimalCeil.h
  DecimalUtil.h
  Factorial.h
  GetJsonObject.h
  Hash.h
  In.h
  InitcapFunction.h
  JsonObjectKeys.h
  LambdaIndexUtil.h
  LeastGreatest.h
  LuhnCheckFunction.h
  MaskFunction.h
  MightContain.h
  MonotonicallyIncreasingId.h
  RaiseError.h
  Rand.h
  RandStr.h
  ReadSidePaddingFunction.h
  RegexFunctions.h
  SimpleComparisonMatcher.h
  Size.h
  SparkPartitionId.h
  Split.h
  String.h
  StringToMap.h
  TimestampUtils.h
  ToJson.h
  ToPrettyString.h
  UnBase64Function.h
  UnscaledValueFunction.h
  Uuid.h
  VarcharTypeWriteSideCheck.h
)

velox_link_libraries(
  velox_functions_spark_impl
  velox_functions_lib
  velox_functions_prestosql_impl
  velox_functions_spark_specialforms
  velox_functions_util
  Folly::folly
  simdjson::simdjson
)

if(NOT VELOX_MONO_LIBRARY)
  set_property(TARGET velox_functions_spark_impl PROPERTY JOB_POOL_COMPILE high_memory_pool)
endif()

add_subdirectory(window)

if(${VELOX_ENABLE_AGGREGATES})
  add_subdirectory(aggregates)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
  add_subdirectory(coverage)
  add_subdirectory(fuzzer)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

add_subdirectory(registration)
add_subdirectory(types)
