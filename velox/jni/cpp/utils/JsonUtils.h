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

#include "folly/json/json.h"
namespace facebook::velox::sdk::utils {
class JsonUtils {
 public:
  static const folly::json::serialization_opts& getOpts() {
    static const folly::json::serialization_opts opts_ = []() {
      folly::json::serialization_opts opts;
      opts.sort_keys = true;
      return opts;
    }();
    return opts_;
  }
  inline static std::string toSortedJson(folly::dynamic dynamic) {
    return serialize(dynamic, getOpts());
  }
};
} // namespace facebook::velox::sdk::utils
