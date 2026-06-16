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
#include "velox/exec/IndexLookupJoinBridge.h"

namespace facebook::velox::exec {

void IndexLookupJoinBridge::setIndexSplits(
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(started_, "Bridge must be started before setting index splits");
    VELOX_CHECK(
        !cancelled_, "Setting index splits after the bridge is cancelled");
    VELOX_CHECK(!splitsSet_, "setIndexSplits must be called only once");
    splitsSet_ = true;
    indexSplits_ = std::move(splits);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
IndexLookupJoinBridge::splitsOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(
      !cancelled_, "Getting index splits after the bridge is cancelled");
  if (splitsSet_) {
    return indexSplits_;
  }
  promises_.emplace_back("IndexLookupJoinBridge::splitsOrFuture");
  *future = promises_.back().getSemiFuture();
  return {};
}

} // namespace facebook::velox::exec
