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
#include "MockExchangeSource.h"
#include <iostream>
#include "velox/exec/Exchange.h"

using namespace facebook::velox;

std::vector<std::string> MockExchangeSource::closedTasks_;

bool MockExchangeSource::shouldRequestLocked() {
  if (atEnd_) {
    return false;
  }
  bool pending = requestPending_;
  requestPending_ = true;
  return !pending;
}

void MockExchangeSource::request() {}

void MockExchangeSource::close() {
  closedTasks_.push_back(this->taskId_);
}

void MockExchangeSource::resetClosedTasks() {
  closedTasks_.clear();
}

bool MockExchangeSource::isTaskClosed(std::string taskId) {
  return std::find(closedTasks_.begin(), closedTasks_.end(), taskId) !=
      closedTasks_.end();
}

std::unique_ptr<exec::ExchangeSource> MockExchangeSource::createExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<exec::ExchangeQueue> queue) {
  if (strncmp(taskId.c_str(), "mock://", 7) == 0) {
    return std::make_unique<MockExchangeSource>(
        taskId, destination, std::move(queue));
  }
  return nullptr;
}