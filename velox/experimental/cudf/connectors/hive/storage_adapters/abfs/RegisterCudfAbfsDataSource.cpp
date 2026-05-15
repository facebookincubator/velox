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

#include "velox/experimental/cudf/connectors/hive/storage_adapters/abfs/RegisterCudfAbfsDataSource.h"

#ifdef VELOX_ENABLE_ABFS
#include "velox/experimental/cudf/connectors/hive/storage_adapters/CudfDataSourceRegistry.h"
#include "velox/experimental/cudf/connectors/hive/storage_adapters/abfs/CudfAbfsDataSource.h"

#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"

#include <folly/synchronization/CallOnce.h>
#endif

namespace facebook::velox::cudf_velox::filesystems {

#ifdef VELOX_ENABLE_ABFS
namespace {
folly::once_flag cudfAbfsRegistrationFlag;
} // namespace
#endif

void registerCudfAbfsDataSource() {
#ifdef VELOX_ENABLE_ABFS
  folly::call_once(cudfAbfsRegistrationFlag, []() {
    registerCudfDataSource(
        [](std::string_view path) {
          return ::facebook::velox::filesystems::isAbfsFile(path);
        },
        [](std::string_view path,
           const std::shared_ptr<const config::ConfigBase>& properties)
            -> std::shared_ptr<cudf::io::datasource> {
          return std::make_shared<CudfAbfsDataSource>(path, properties);
        });
  });
#endif
}

} // namespace facebook::velox::cudf_velox::filesystems
