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
#include <folly/Uri.h>

#include "velox/exec/Exchange.h"

using namespace facebook::velox;

class MockExchangeSource : public exec::ExchangeSource {
 public:
  MockExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue)
      : ExchangeSource(taskId, destination, queue) {}
  bool shouldRequestLocked() override;
  static std::unique_ptr<ExchangeSource> createExchangeSource(
      const std::string& url,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue);

  static void resetClosedTasks();
  static bool isTaskClosed(std::string taskId);

 private:
  void request() override;
  void close() override;
  static std::vector<std::string> closedTasks_;
};
