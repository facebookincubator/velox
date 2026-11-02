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

#include <optional>

#include "velox/common/base/Exceptions.h"
#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::exec {

class JoinBridge {
 public:
  virtual ~JoinBridge() = default;

  /// Invoked by the associated task after the driver creations to start this
  /// join bridge before starting task execution.
  virtual void start();

  /// Sets this to a cancelled state and unblocks any waiting activity. This may
  /// happen asynchronously before or after the result has been set.
  void cancel();

 protected:
  static void notify(std::vector<ContinuePromise> promises);

  std::mutex mutex_;
  bool started_{false};
  std::vector<ContinuePromise> promises_;
  bool cancelled_{false};
};

/// Hands the build side of a join, built once by the last build driver, to the
/// probe drivers.
template <typename T>
class BuildResultBridge : public JoinBridge {
 public:
  /// Called once; wakes up the waiting probe drivers.
  void setData(T data) {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      VELOX_CHECK(!data_.has_value(), "setData must be called only once");
      data_ = std::move(data);
      promises = std::move(promises_);
    }
    notify(std::move(promises));
  }

  /// Returns the build side, or std::nullopt and sets 'future' if it is not
  /// built yet.
  std::optional<T> dataOrFuture(ContinueFuture* future) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
    if (data_.has_value()) {
      return data_;
    }
    promises_.emplace_back("BuildResultBridge::dataOrFuture");
    *future = promises_.back().getSemiFuture();
    return std::nullopt;
  }

 private:
  std::optional<T> data_;
};
} // namespace facebook::velox::exec
