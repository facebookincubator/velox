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

#include <optional>
#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"

namespace facebook::velox::connector::tpch {

/// Tpch connector configs.
class TpchConfig {
 public:
  static constexpr const char* kTextPoolSizeMb = "tpch.text-pool-size-mb";

  int32_t textPoolSizeMb() const;

  TpchConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for TpchConfig initialization");
    config_ = std::move(config);
  }

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

 private:
  std::shared_ptr<const config::ConfigBase> config_;
};

} // namespace facebook::velox::connector::tpch
