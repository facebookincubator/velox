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
#pragma once

namespace facebook::velox::connector {

// TODO: Add a demo connector plugin

constexpr const char* kFuzzerConnectorName  = "fuzzer";
constexpr const char* kHiveConnectorName    = "hive";
constexpr const char* kHiveV2ConnectorName  = "hive_v2";
constexpr const char* kTpchConnectorName    = "tpch";

// Define the Hive ColumnType as strings to avoid direct Hive reference
inline constexpr const char* kColumnTypeRegular = "regular";
inline constexpr const char* kColumnTypePartition = "partition_key";
inline constexpr const char* kColumnTypeSynthesized = "synthesized";

}