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

#include <algorithm>
#include <utility>

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/experimental/cudf/CudfConfig.h"

namespace facebook::velox::ucx_exchange {

/// Bounded executor for exchange codec work.
///
/// The UCXX worker has one progress thread. Running a synchronous GPU codec on
/// that thread stalls every connection in the process. This executor lets the
/// progress thread continue while a deliberately small number of codec tasks
/// are active. One thread is the default because a codec kernel already uses
/// most SMs; additional threads are an explicit tuning parameter.
inline bool codecPipelineEnabled() {
  return cudf_velox::CudfConfig::getInstance().exchangeCompressionPipeline;
}

inline std::size_t codecPipelineThreads() {
  return static_cast<std::size_t>(std::clamp(
      cudf_velox::CudfConfig::getInstance().exchangeCompressionPipelineThreads,
      1,
      4));
}

inline folly::CPUThreadPoolExecutor& codecPipelineExecutor() {
  // Worker-lifetime singleton. Intentionally leaked to avoid static-destruction
  // ordering between Folly worker threads, RMM, and the CUDA runtime at process
  // shutdown.
  static auto* executor =
      new folly::CPUThreadPoolExecutor(codecPipelineThreads());
  return *executor;
}

template <typename Task>
inline void submitCodecTask(Task&& task) {
  codecPipelineExecutor().add(std::forward<Task>(task));
}

} // namespace facebook::velox::ucx_exchange
