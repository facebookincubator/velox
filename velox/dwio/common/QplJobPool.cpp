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

#include "velox/dwio/common/QplJobPool.h"
#include <folly/Random.h>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {

std::array<std::atomic<bool>, QplJobHWPool::MAX_JOB_NUMBER>
    QplJobHWPool::hwJobPtrLocks;

QplJobHWPool& QplJobHWPool::getInstance() {
  static QplJobHWPool pool;
  return pool;
}

QplJobHWPool::QplJobHWPool() {
  if (!iaaJobReady) {
    allocateQPLJob();
  }
}

QplJobHWPool::~QplJobHWPool() {
  for (uint32_t i = 0; i < MAX_JOB_NUMBER; ++i) {
    if (hwJobPtrPool[i]) {
      qpl_fini_job(hwJobPtrPool[i]);
      hwJobPtrPool[i] = nullptr;
    }
  }
  iaaJobReady = false;
}

/**
 * Allocate qpl job and put it into hwJobPtrPool
 */
void QplJobHWPool::allocateQPLJob() {
  uint32_t job_size = 0;

  // Get size required for saving a single qpl job object
  qpl_get_job_size(qpl_path, &job_size);

  // Allocate entire buffer for storing all job objects
  hwJobsBuffer = std::make_unique<uint8_t[]>(job_size * MAX_JOB_NUMBER);

  // Initialize pool for storing all job object pointers
  // Allocate buffer by shifting address offset for each job object.
  hwJobPtrPool.resize(MAX_JOB_NUMBER);
  for (uint32_t i = 0; i < MAX_JOB_NUMBER; ++i) {
    qpl_job* qplJobPtr =
        reinterpret_cast<qpl_job*>(hwJobsBuffer.get() + i * job_size);
    auto status = qpl_init_job(qpl_path, qplJobPtr);
    if (status != QPL_STS_OK) {
      iaaJobReady = false;
      LOG(WARNING) << "Initialization of hardware IAA failed, statsu: "
                   << status << ". Please check if Intel \
        In-Memory Analytics Accelerator (IAA) is properly set up!";
      return;
    }
    this->hwJobPtrPool[i] = qplJobPtr;
    hwJobPtrLocks[i].store(false);
  }

  iaaJobReady = true;
  return;
}

/**
 * Acquire a deflate job.
 * QplJobHWPool maintains MAX_JOB_NUMBER job slot to avoid frequently allocate,
 * initialize and release job. Random slots is used to select a job and
 * tryLockJob will check if the job is free.
 * @return job_id and qpl_job pair
 */
std::pair<int, qpl_job*> QplJobHWPool::acquireDeflateJob() {
  std::pair<int, qpl_job*> res;
  res.first = -1;
  res.second = nullptr;
  if (!job_ready()) {
    return res;
  }
  uint32_t retry = 0;
  uint32_t index = folly::Random::rand32(1, MAX_JOB_NUMBER - 1);
  while (!tryLockJob(index)) {
    index = folly::Random::rand32(1, MAX_JOB_NUMBER - 1);
    retry++;
    if (retry > MAX_JOB_NUMBER) {
      return res;
    }
  }
  res.first = index;
  if (index >= MAX_JOB_NUMBER) {
    return res;
  }
  res.second = hwJobPtrPool[index];

  return res;
}

void QplJobHWPool::releaseJob(int job_id) {
  if (job_id >= MAX_JOB_NUMBER || job_id <= 0) {
    return;
  }
  VELOX_CHECK_LT(job_id, MAX_JOB_NUMBER);
  hwJobPtrLocks[job_id].store(false);
  return;
}

bool QplJobHWPool::tryLockJob(uint32_t index) {
  bool expected = false;
  VELOX_CHECK_LT(index, MAX_JOB_NUMBER);
  return hwJobPtrLocks[index].compare_exchange_strong(expected, true);
}

} // namespace facebook::velox::dwio::common
