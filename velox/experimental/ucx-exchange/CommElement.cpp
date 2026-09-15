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

#include "velox/experimental/ucx-exchange/CommElement.h"

#include "velox/experimental/ucx-exchange/Communicator.h"

namespace facebook::velox::ucx_exchange {

bool CommElement::enqueueStateEventImpl(
    std::shared_ptr<CommElement> self,
    StateEvent event) noexcept {
  try {
    std::lock_guard<std::mutex> lock(stateEventMutex_);
    stateEvents_.push_back(std::move(event));
    try {
      // Keep the event queue locked through work-queue insertion. If insertion
      // cannot allocate, removing the event is safe because drainStateEvents()
      // cannot race and consume it in between.
      communicator_->addToWorkQueue(std::move(self));
    } catch (...) {
      stateEvents_.pop_back();
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool CommElement::drainStateEvents() {
  std::deque<StateEvent> events;
  {
    std::lock_guard<std::mutex> lock(stateEventMutex_);
    events.swap(stateEvents_);
  }

  for (auto& event : events) {
    event();
  }
  return !events.empty();
}

} // namespace facebook::velox::ucx_exchange
