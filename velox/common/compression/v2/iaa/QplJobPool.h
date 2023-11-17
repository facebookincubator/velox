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

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <qpl/qpl.h>

namespace facebook::velox::common::iaa {

static constexpr auto kMaxQplJobNumber = 1024;

/// QplJobHWPool is resource pool to provide the job objects, which is
/// used for storing context information during.
/// Memory for QPL job will be allocated when the QPLJobHWPool instance is
/// created.
/// QPL job can offload RLE-decoding/Filter/(De)compression works to hardware
/// accelerator.
class QplJobHWPool {
 public:
  static QplJobHWPool& getInstance();

  /// Setup maximum job number.
  /// Must be called before first time calling getInstance().
  static void initialize(uint32_t maxJobNumber);

  /// Acquire QPL job.
  /// \param jobId QPL job id, used when release QPL job
  /// \return Pointer to the QPL job. If failed, return nullptr.
  qpl_job* acquireJob(uint32_t& jobId);

  // Release QPL job by the jobId.
  void releaseJob(uint32_t jobId);

  // Return if the QPL job is allocated successfully.
  const bool& jobPoolReady() const;

  uint32_t maxJobNumber();

  // Deleted copy constructor and assignment operator.
  QplJobHWPool(const QplJobHWPool&) = delete;
  QplJobHWPool& operator=(const QplJobHWPool&) = delete;

 private:
  // Max jobs in jobPool. Entire buffer for storing all job objects.
  std::vector<uint8_t> hwJobsBuffer_;
  // Job pool for storing all job object pointers.
  std::array<qpl_job*, kMaxQplJobNumber> jobPool_;
  // Locks for accessing each job object pointers.
  std::array<std::atomic_bool, kMaxQplJobNumber> jobLocks_;

  bool jobPoolReady_{false};
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<uint32_t> distribution_;

  static uint32_t maxJobNumber_;
  static std::once_flag initFlag_;
  static bool initialized_;

  QplJobHWPool();
  ~QplJobHWPool();

  void initJobPool();
  bool tryLockJob(uint32_t index);
  void unLockJob(uint32_t index);
};

} // namespace facebook::velox::common::iaa
