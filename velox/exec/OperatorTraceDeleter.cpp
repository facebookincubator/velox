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
#include "velox/exec/OperatorTraceDeleter.h"

#include <folly/FileUtil.h>
#include <filesystem>

namespace facebook::velox::exec::trace {
OperatorTraceDeleter::OperatorTraceDeleter() {
  deleteFileExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(
      1, std::make_shared<folly::NamedThreadFactory>("QueryDataDeleter"));
}

void OperatorTraceDeleter::asyncDeleteDir(const std::string& dirPath) {
  if (std::filesystem::exists(dirPath)) {
    deleteFileExecutor_->add([dirPath] {
      if (std::filesystem::exists(dirPath)) {
        LOG(INFO) << "QueryDataDeleter removing all files from " << dirPath;
        try {
          auto n = std::filesystem::remove_all(dirPath);
          LOG(INFO) << "QueryDataDeleter has removed " << n << " files.";
        } catch (...) {
          // Since the deletion of the directory is performed asynchronously,
          // the exception thrown will not be received by the outside, and this
          // exception has no other bad impact, so it can be ignored directly.
          LOG(WARNING) << "QueryDataDeleter failed to remove all trace files.";
        }
      }
    });
  }
}

OperatorTraceDeleter* OperatorTraceDeleter::instance() {
  static std::unique_ptr<OperatorTraceDeleter> instance =
      std::make_unique<OperatorTraceDeleter>();
  return instance.get();
}
} // namespace facebook::velox::exec::trace
