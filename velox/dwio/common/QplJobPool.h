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

#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "qpl/qpl.h"

namespace facebook::velox::dwio::common {

// QplJobHWPool is resource pool to provide the job that will be submitted to
// Intel® IAA Memory for Intel® IAA job will be allocated when the QPLJobHWPool
// instance is created.
//
// The Intel® In-Memory Analytics Accelerator (Intel® IAA) is a hardware
// accelerator that provides very high throughput compression and decompression
// combined with primitive analytic functions. It primarily targets applications
// such as big-data and in-memory analytic databases, as well as
// application-transparent usages such as memory page compression. Intel® IAA
// contains two main functional blocks: Compression and Analytics. The Analytics
// pipe contains two sub-blocks: Decompress and Filter. These functions are tied
// together, so that each analytics operation can perform decompress-only,
// filter-only, or decompress-and-filter processing.
//
// Intel QPL is library to provide application programming interface (API) for
// interaction with Intel® In-Memory Analytics Accelerator (Intel® IAA) hardware
//
// Intel® IAA:
// https://www.intel.com/content/www/us/en/content-details/780887/intel-in-memory-analytics-accelerator-intel-iaa.html
// Intel QPL:
// https://intel.github.io/qpl/documentation/introduction_docs/introduction.html
class QplJobHWPool {
 public:
  static QplJobHWPool& getInstance();
  QplJobHWPool();
  ~QplJobHWPool();

  // Release QPL job by the job_id.
  void releaseJob(int job_id);

  // Return if the QPL job is allocated sucessfully.
  const bool& job_ready() {
    return iaaJobReady;
  }

  std::pair<int, qpl_job*> acquireDeflateJob();

  /**
   * Get qpl job by job id
   * @param job_id the job id or index in the qpl job pool
   * @return nullptr if the job id is invalid
   */
  qpl_job* getJobById(int job_id) {
    if (job_id >= MAX_JOB_NUMBER || job_id <= 0) {
      return nullptr;
    }
    return hwJobPtrPool[job_id];
  }

  static constexpr uint32_t MAX_JOB_NUMBER = 1024;

 private:
  bool tryLockJob(uint32_t index);
  void allocateQPLJob();

  qpl_path_t qpl_path = qpl_path_hardware;

  // Max jobs in QPL_JOB_POOL
  // Entire buffer for storing all job objects
  std::unique_ptr<uint8_t[]> hwJobsBuffer;

  // Job pool for storing all job object pointers
  std::vector<qpl_job*> hwJobPtrPool;

  // Locks for accessing each job object pointers
  bool iaaJobReady;
  static std::array<std::atomic<bool>, MAX_JOB_NUMBER> hwJobPtrLocks;
};

} // namespace facebook::velox::dwio::common
