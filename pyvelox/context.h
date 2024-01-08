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

#include <pybind11/stl.h>
#include <velox/common/file/FileSystems.h>
#include <velox/common/memory/Memory.h>
#include <velox/connectors/hive/HiveConnector.h>
#include <velox/core/QueryCtx.h>
#include <velox/dwio/dwrf/reader/DwrfReader.h>

namespace facebook::velox::py {

/// PyVeloxContext is used only during function binding time. Its a utility
/// that manages pool, query and exec context for Velox expressions and vectors.
struct PyVeloxContext {
  static inline PyVeloxContext& getSingletonInstance() {
    if (!instance_) {
      instance_ = std::unique_ptr<PyVeloxContext>(new PyVeloxContext());
    }
    return *instance_.get();
  }

  facebook::velox::memory::MemoryPool* pool() {
    return pool_.get();
  }

  facebook::velox::core::QueryCtx* queryCtx() {
    return queryCtx_.get();
  }

  facebook::velox::core::ExecCtx* execCtx() {
    return execCtx_.get();
  }

  static inline void cleanup() {
    if (instance_) {
      instance_.reset();
    }
  }

 private:
  PyVeloxContext() = default;
  PyVeloxContext(const PyVeloxContext&) = delete;
  PyVeloxContext(const PyVeloxContext&&) = delete;
  PyVeloxContext& operator=(const PyVeloxContext&) = delete;
  PyVeloxContext& operator=(const PyVeloxContext&&) = delete;

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_ =
      facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  std::shared_ptr<facebook::velox::core::QueryCtx> queryCtx_ =
      std::make_shared<facebook::velox::core::QueryCtx>();
  std::unique_ptr<facebook::velox::core::ExecCtx> execCtx_ =
      std::make_unique<facebook::velox::core::ExecCtx>(
          pool_.get(),
          queryCtx_.get());

  static inline std::unique_ptr<PyVeloxContext> instance_;
};

struct PySubstraitContext {
  PySubstraitContext() = default;
  PySubstraitContext(const PySubstraitContext&) = delete;
  PySubstraitContext(const PySubstraitContext&&) = delete;
  PySubstraitContext& operator=(const PySubstraitContext&) = delete;
  PySubstraitContext& operator=(const PySubstraitContext&&) = delete;

  static inline PySubstraitContext& getInstance() {
    if (!instance_) {
      instance_ = std::make_unique<PySubstraitContext>();
    }
    return *instance_.get();
  }

  inline void initialize() {
    facebook::velox::connector::registerConnector(connector_);
    filesystems::registerLocalFileSystem();
    dwrf::registerDwrfReaderFactory();
  }

  inline void finalize() {
    facebook::velox::connector::unregisterConnector(kConnectorId);
    dwrf::unregisterDwrfReaderFactory();
  }

  static inline void cleanup() {
    if (instance_) {
      instance_.reset();
    }
  }

 private:
  const std::string kConnectorId =
      "test-hive"; // same name used in SubstraitVeloxConverter
  std::shared_ptr<facebook::velox::connector::Connector> connector_ =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(
              kConnectorId,
              std::make_shared<facebook::velox::core::MemConfig>());

  static inline std::unique_ptr<PySubstraitContext> instance_;
};

} // namespace facebook::velox::py
