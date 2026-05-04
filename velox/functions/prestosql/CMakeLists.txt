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
add_subdirectory(json)
add_subdirectory(types)
add_subdirectory(window)

velox_add_library(
  velox_functions_prestosql_impl
  ArrayConstructor.cpp
  ArrayContains.cpp
  ArrayDistinct.cpp
  ArrayDuplicates.cpp
  ArrayFlatten.cpp
  ArrayIntersectExcept.cpp
  ArrayAndMapMatch.cpp
  ArrayMinMaxBy.cpp
  ArrayPosition.cpp
  ArraySort.cpp
  ArraySum.cpp
  Comparisons.cpp
  DecimalFunctions.cpp
  ElementAt.cpp
  FilterFunctions.cpp
  FindFirst.cpp
  FromUtf8.cpp
  InPredicate.cpp
  JsonFunctions.cpp
  Map.cpp
  MapEntries.cpp
  MapKeysAndValues.cpp
  MapSubset.cpp
  MapZipWith.cpp
  Not.cpp
  Reduce.cpp
  Reverse.cpp
  RowFunction.cpp
  Sequence.cpp
  SfmSketchFunctions.cpp
  Split.cpp
  SplitToMap.cpp
  StringFunctions.cpp
  Subscript.cpp
  ToUtf8.cpp
  Transform.cpp
  TransformWithIndex.cpp
  TransformKeys.cpp
  TransformValues.cpp
  TypeOf.cpp
  UpperLower.cpp
  URIParser.cpp
  VectorArithmetic.cpp
  WidthBucketArray.cpp
  Zip.cpp
  ZipWith.cpp
  HEADERS
  Arithmetic.h
  ArithmeticImpl.h
  ArrayConstructor.h
  ArrayFunctions.h
  ArraySort.h
  ArraySubset.h
  BinaryFunctions.h
  BingTileFunctions.h
  Bitwise.h
  Cardinality.h
  Comparisons.h
  DataSizeFunctions.h
  DateTimeFunctions.h
  DateTimeImpl.h
  DecimalFunctions.h
  DistanceFunctions.h
  DotProduct.h
  EnumFunctions.h
  Fail.h
  FloatingPointFunctions.h
  GeometryFunctions.h
  GreatestLeast.h
  HyperLogLogFunctions.h
  IPAddressFunctions.h
  InPredicate.h
  IntegerFunctions.h
  JsonFunctions.h
  KHyperLogLogFunctions.h
  L2Norm.h
  Map.h
  MapAppend.h
  MapExcept.h
  MapFunctions.h
  MapIntersect.h
  MapKeysByTopNValues.h
  MapKeysOverlap.h
  MapNormalize.h
  MapSubset.h
  MapTopN.h
  MapTopNImpl.h
  MapTopNKeys.h
  MapTopNValues.h
  MapTrimValues.h
  MapUpdate.h
  MapValuesInRange.h
  MultimapFromEntries.h
  Probability.h
  QDigestFunctions.h
  Rand.h
  Reduce.h
  RegexpReplace.h
  RegexpSplit.h
  RemapKeys.h
  S2Functions.h
  SetDigestFunctions.h
  SfmSketchFunctions.h
  SplitPart.h
  SplitToMap.h
  SplitToMultiMap.h
  StringFunctions.h
  TDigestFunctions.h
  URIParser.h
  URLFunctions.h
  UuidFunctions.h
  WidthBucketArray.h
  WordStem.h
)

velox_link_libraries(
  velox_functions_prestosql_impl
  velox_common_hyperloglog
  velox_functions_json
  velox_functions_lib
  velox_expression
  velox_external_md5
  velox_type_tz
  velox_presto_types
  velox_functions_util
  Folly::folly
  FastFloat::fast_float
  stemmer::stemmer
)

if(VELOX_ENABLE_FAISS)
  velox_link_libraries(velox_functions_prestosql_impl FAISS::faiss)
  velox_compile_definitions(velox_functions_prestosql_impl PRIVATE VELOX_ENABLE_FAISS)
endif()

if(VELOX_ENABLE_GEO)
  add_subdirectory(geospatial)

  # S2 cell operations are isolated in a separate library to prevent a
  # nallocx symbol conflict between s2geometry's malloc_extension.h and
  # folly's MallocImpl.h.
  velox_add_library(velox_s2_cell_operations S2CellOperations.cpp HEADERS S2CellOperations.h)
  velox_link_libraries(velox_s2_cell_operations s2::s2 GEOS::geos)

  velox_link_libraries(velox_functions_prestosql_impl velox_functions_geo velox_s2_cell_operations)
endif()

if(NOT VELOX_MONO_LIBRARY)
  set_property(TARGET velox_functions_prestosql_impl PROPERTY JOB_POOL_COMPILE high_memory_pool)
endif()

if(${VELOX_ENABLE_AGGREGATES})
  add_subdirectory(aggregates)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
  add_subdirectory(coverage)
  add_subdirectory(fuzzer)
elseif(${VELOX_BUILD_TEST_UTILS})
  add_subdirectory(tests/utils)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

add_subdirectory(registration)
