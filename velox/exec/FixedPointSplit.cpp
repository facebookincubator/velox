/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include "velox/exec/FixedPointSplit.h"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/container/F14Set.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

FixedPointSplit::FixedPointSplit(std::vector<std::string> workers)
    : ConnectorSplit(""), workers(std::move(workers)) {
  VELOX_USER_CHECK(!this->workers.empty(), "FixedPointSplit names no workers");
  folly::F14FastSet<std::string_view> names;
  for (const auto& worker : this->workers) {
    VELOX_USER_CHECK(!worker.empty(), "Empty worker name in FixedPointSplit");
    VELOX_USER_CHECK(
        names.insert(worker).second,
        "Duplicate worker in FixedPointSplit: {}",
        worker);
  }
}

folly::dynamic FixedPointSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "FixedPointSplit";
  folly::dynamic workerNames = folly::dynamic::array;
  for (const auto& worker : workers) {
    workerNames.push_back(worker);
  }
  obj["workers"] = std::move(workerNames);
  return obj;
}

// static
std::shared_ptr<FixedPointSplit> FixedPointSplit::create(
    const folly::dynamic& obj) {
  std::vector<std::string> workers;
  for (const auto& worker : obj["workers"]) {
    workers.push_back(worker.asString());
  }
  return std::make_shared<FixedPointSplit>(std::move(workers));
}

// static
void FixedPointSplit::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("FixedPointSplit", FixedPointSplit::create);
}

std::string FixedPointSplit::toString() const {
  return fmt::format("FixedPoint: [{}]", fmt::join(workers, ", "));
}

} // namespace facebook::velox::exec
