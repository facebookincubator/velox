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

#include <glog/logging.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/v2/iaa/QplJobPool.h"

namespace facebook::velox::common::iaa {

namespace {
inline void checkJobIndex(uint32_t index) {
  auto maxJobNumber = QplJobHWPool::getInstance().maxJobNumber();
  if (index >= maxJobNumber) {
    VELOX_FAIL(
        "Job index {} overflow. Must be in range (0, {})", index, maxJobNumber);
  }
}
} // namespace

uint32_t QplJobHWPool::maxJobNumber_;
std::once_flag QplJobHWPool::initFlag_;
bool QplJobHWPool::initialized_;

void QplJobHWPool::initialize(uint32_t maxJobNumber) {
  std::call_once(initFlag_, [&] {
    if (maxJobNumber > kMaxQplJobNumber) {
      VELOX_FAIL(
          "Job number {} exceeds maximum job number {}",
          maxJobNumber,
          kMaxQplJobNumber);
    }
    maxJobNumber_ = maxJobNumber;
    initialized_ = true;
  });
}

QplJobHWPool& QplJobHWPool::getInstance() {
  std::call_once(initFlag_, [] {
    if (!initialized_) {
      VELOX_FAIL("QplJobHWPool not initialized");
    }
  });
  static QplJobHWPool pool;
  return pool;
}

QplJobHWPool::QplJobHWPool()
    : randomEngine_(std::random_device()()),
      distribution_(0, maxJobNumber_ - 1) {
  initJobPool();
}

QplJobHWPool::~QplJobHWPool() {
  for (auto i = 0; i < maxJobNumber_; ++i) {
    if (jobPool_[i]) {
      while (!tryLockJob(i))
        ;
      qpl_fini_job(jobPool_[i]);
      unLockJob(i);
      jobPool_[i] = nullptr;
    }
  }
}

void QplJobHWPool::initJobPool() {
  distribution_ = std::uniform_int_distribution<uint32_t>(0, maxJobNumber_ - 1);

  uint32_t jobSize = 0;
  const char* qpl_version = qpl_get_library_version();

  // Get size required for saving a single qpl job object.
  qpl_get_job_size(qpl_path_hardware, &jobSize);
  // Allocate entire buffer for storing all job objects.
  hwJobsBuffer_.resize(jobSize * maxJobNumber_);
  // Initialize pool for storing all job object pointers.
  // Reallocate buffer by shifting address offset for each job object.
  for (uint32_t index = 0; index < maxJobNumber_; ++index) {
    qpl_job* qplJobPtr =
        reinterpret_cast<qpl_job*>(hwJobsBuffer_[index * jobSize]);
    if (auto status = qpl_init_job(qpl_path_hardware, qplJobPtr);
        status != QPL_STS_OK) {
      jobPoolReady_ = false;
      VLOG(1)
          << "Initialization of hardware-assisted Qpl codec failed at index: "
          << index << ", falling back to SW codec. "
          << "(Details: QplJobHWPool->qpl_init_job failed, status: " << status
          << " - Please refer to qpl_status in "
          << "./contrib/qpl/include/qpl/c_api/status.h). "
          << "Please check if Intel In-Memory Analytics Accelerator (IAA) "
          << "is properly set up. QPL Version: " << qpl_version;
      return;
    }
    jobPool_[index] = qplJobPtr;
    jobLocks_[index].store(false);
  }
  jobPoolReady_ = true;
}

qpl_job* QplJobHWPool::acquireJob(uint32_t& jobId) {
  if (!jobPoolReady()) {
    return nullptr;
  }
  uint32_t retry = 0;
  auto index = distribution_(randomEngine_);
  while (!tryLockJob(index)) {
    index = distribution_(randomEngine_);
    retry++;
    if (retry > maxJobNumber_) {
      return nullptr;
    }
  }
  jobId = maxJobNumber_ - index;
  return jobPool_[index];
}

void QplJobHWPool::releaseJob(uint32_t jobId) {
  if (jobPoolReady()) {
    auto index = maxJobNumber_ - jobId;
    unLockJob(index);
  }
}

bool QplJobHWPool::tryLockJob(uint32_t index) {
  checkJobIndex(index);
  bool expected = false;
  return jobLocks_[index].compare_exchange_strong(expected, true);
}

void QplJobHWPool::unLockJob(uint32_t index) {
  checkJobIndex(index);
  jobLocks_[index].store(false);
}

const bool& QplJobHWPool::jobPoolReady() const {
  return jobPoolReady_;
}

uint32_t QplJobHWPool::maxJobNumber() {
  return maxJobNumber_;
}

} // namespace facebook::velox::common::iaa
