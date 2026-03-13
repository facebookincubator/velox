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

#include "velox/connectors/hive/HiveWriterTypes.h"

namespace facebook::velox::connector::hive {

const HiveWriterId& HiveWriterId::unpartitionedId() {
  static const HiveWriterId writerId{0};
  return writerId;
}

std::string HiveWriterId::toString() const {
  if (partitionId.has_value() && bucketId.has_value()) {
    return fmt::format("part[{}.{}]", partitionId.value(), bucketId.value());
  }

  if (partitionId.has_value() && !bucketId.has_value()) {
    return fmt::format("part[{}]", partitionId.value());
  }

  // This WriterId is used to add an identifier in the MemoryPools. This could
  // indicate unpart, but the bucket number needs to be disambiguated. So
  // creating a new label using bucket.
  if (!partitionId.has_value() && bucketId.has_value()) {
    return fmt::format("bucket[{}]", bucketId.value());
  }

  return "unpart";
}

} // namespace facebook::velox::connector::hive
