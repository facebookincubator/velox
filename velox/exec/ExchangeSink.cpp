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

#include "velox/exec/ExchangeSink.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

void ExchangeSink::append(
    int32_t partition,
    std::unique_ptr<folly::IOBuf> data) {
  VELOX_CHECK_NOT_NULL(data);
  const auto range = data->coalesce();
  append(
      partition,
      std::string_view(
          reinterpret_cast<const char*>(range.data()), range.size()));
}

void ExchangeSink::appendBatch(
    int32_t partition,
    std::deque<std::unique_ptr<folly::IOBuf>> data) {
  VELOX_CHECK(!data.empty());
  auto chain = std::move(data.front());
  data.pop_front();
  for (auto& buffer : data) {
    chain->appendToChain(std::move(buffer));
  }
  append(partition, std::move(chain));
}

} // namespace facebook::velox::exec
