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

add_subdirectory(parser)

velox_add_library(
  velox_presto_types
  BigintEnumRegistration.cpp
  BigintEnumType.cpp
  PrestoTypes.cpp
  BingTileRegistration.cpp
  BingTileType.cpp
  EnumTypeBase.cpp
  HyperLogLogRegistration.cpp
  IPAddressRegistration.cpp
  IPPrefixRegistration.cpp
  JsonCastOperator.cpp
  JsonRegistration.cpp
  KHyperLogLogRegistration.cpp
  QDigestRegistration.cpp
  P4HyperLogLogRegistration.cpp
  SfmSketchRegistration.cpp
  TDigestRegistration.cpp
  TimestampWithTimeZoneRegistration.cpp
  TimestampWithTimeZoneType.cpp
  TimeWithTimezoneRegistration.cpp
  TimeWithTimezoneType.cpp
  IPAddressType.cpp
  IPPrefixType.cpp
  SetDigestRegistration.cpp
  UuidRegistration.cpp
  UuidType.cpp
  VarcharEnumRegistration.cpp
  VarcharEnumType.cpp
  HEADERS
  BigintEnumRegistration.h
  BigintEnumType.h
  BingTileRegistration.h
  BingTileType.h
  EnumTypeBase.h
  GeometryRegistration.h
  GeometryType.h
  HyperLogLogRegistration.h
  HyperLogLogType.h
  IPAddressRegistration.h
  IPAddressType.h
  IPPrefixRegistration.h
  IPPrefixType.h
  JsonCastOperator.h
  JsonRegistration.h
  JsonType.h
  KHyperLogLogRegistration.h
  KHyperLogLogType.h
  P4HyperLogLogRegistration.h
  P4HyperLogLogType.h
  PrestoTypes.h
  QDigestRegistration.h
  QDigestType.h
  SetDigestRegistration.h
  SetDigestType.h
  SfmSketchRegistration.h
  SfmSketchType.h
  SphericalGeographyRegistration.h
  SphericalGeographyType.h
  TDigestRegistration.h
  TDigestType.h
  TimeWithTimezoneRegistration.h
  TimeWithTimezoneType.h
  TimestampWithTimeZoneRegistration.h
  TimestampWithTimeZoneType.h
  UuidRegistration.h
  UuidType.h
  VarcharEnumRegistration.h
  VarcharEnumType.h
)
if(VELOX_ENABLE_GEO)
  velox_sources(
    velox_presto_types
    PRIVATE GeometryRegistration.cpp SphericalGeographyRegistration.cpp
  )
endif()

velox_link_libraries(
  velox_presto_types
  velox_type
  velox_memory
  velox_expression
  velox_functions_util
  velox_functions_json
  velox_functions_lib_date_time_formatter
  velox_constrained_input_generators
  velox_presto_types_fuzzer_utils
  velox_common_hyperloglog
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

add_subdirectory(fuzzer_utils)
