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

#include <folly/executors/IOThreadPoolExecutor.h>

namespace facebook::velox::exec::trace {
class OperatorTraceDeleter {
 public:
  OperatorTraceDeleter();
  static OperatorTraceDeleter* instance();

  // On the same node, multiple tasks of one single query may delete old files
  // concurrently, which further leads to deleting failure. To solve this
  // problem, we submit the deletion task to a thread pool with only a single
  // thread, so that all concurrent tasks will be executed sequentially.
  void asyncDeleteDir(const std::string& dirPath);

 private:
  std::unique_ptr<folly::IOThreadPoolExecutor> deleteFileExecutor_;
};
} // namespace facebook::velox::exec::trace
