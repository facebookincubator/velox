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

#include <sstream>
#include <string>
#include <tuple>

namespace facebook::velox::cudf_exchange {

/// @brief A client asks for a partition. The partition is uniquely identified
/// by the taskId and the destination (=partition) number. The partition key
/// implements the C++'s compare requirements
struct PartitionKey {
  std::string taskId;
  uint32_t destination;

  // Less-than operator
  bool operator<(const PartitionKey& other) const {
    return std::tie(taskId, destination) <
        std::tie(other.taskId, other.destination);
  }

  std::string toString() const {
    std::stringstream out;
    out << taskId << "/" << std::to_string(destination);
    return out.str();
  }
};

} // namespace facebook::velox::cudf_exchange
