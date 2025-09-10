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

#include "velox/connectors/RegisterConnectorFactories.h"

#ifdef VELOX_ENABLE_HIVE_CONNECTOR
#include "velox/connectors/hive/HiveConnector.h"
#endif
#ifdef VELOX_ENABLE_TPCDS_CONNECTOR
#include "velox/connectors/tpcds/TpcdsConnector.h"
#endif
#ifdef VELOX_ENABLE_TPCH_CONNECTOR
#include "velox/connectors/tpch/TpchConnector.h"
#endif
#ifdef VELOX_ENABLE_FUZZER_CONNECTOR
#include "velox/connectors/fuzzer/FuzzerConnector.h"
#endif

namespace facebook::velox::connector {

void registerConnectorFactories() {
#ifdef VELOX_ENABLE_HIVE_CONNECTOR
  hive::registerHiveConnectorFactory();
#endif
#ifdef VELOX_ENABLE_TPCDS_CONNECTOR
  tpcds::registerTpcdsConnectorFactory();
#endif
#ifdef VELOX_ENABLE_TPCH_CONNECTOR
  tpch::registerTpchConnectorFactory();
#endif
#ifdef VELOX_ENABLE_FUZZER_CONNECTOR
  fuzzer::registerFuzzerConnectorFactory();
#endif
}

void unregisterConnectorFactories() {
#ifdef VELOX_ENABLE_HIVE_CONNECTOR
  hive::registerHiveConnectorFactory();
#endif
#ifdef VELOX_ENABLE_TPCDS_CONNECTOR
  tpcds::registerTpcdsConnectorFactory();
#endif
#ifdef VELOX_ENABLE_TPCH_CONNECTOR
  tpch::registerTpchConnectorFactory();
#endif
#ifdef VELOX_ENABLE_FUZZER_CONNECTOR
  fuzzer::registerFuzzerConnectorFactory();
#endif
}

} // namespace facebook::velox::connector
