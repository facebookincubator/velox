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

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/QplJobPool.h"
#include "velox/dwio/common/compression/AsyncCompression.h"

namespace facebook::velox::dwio::common::compression {

class GzipIAADecompressor : public AsyncDecompressor {
 public:
  explicit GzipIAADecompressor() {}

  folly::SemiFuture<uint64_t> decompressAsync(
      const char* src,
      uint64_t srcLength,
      char* dest,
      uint64_t destLength) override;

  int waitResult(int job_id);
};

folly::SemiFuture<uint64_t> GzipIAADecompressor::decompressAsync(
    const char* src,
    uint64_t srcLength,
    char* dest,
    uint64_t destLength) {
  dwio::common::QplJobHWPool& qpl_job_pool =
      dwio::common::QplJobHWPool::getInstance();
  auto deflate_job = qpl_job_pool.acquireDeflateJob();
  auto job = deflate_job.second;
  if (job == nullptr) {
    LOG(WARNING) << "cannot AcquireDeflateJob ";
    return folly::makeSemiFutureWith([]() -> uint64_t {
      throw std::runtime_error("Cannot acquire deflate job from pool");
      return 0;
    });
  }
  job->op = qpl_op_decompress;
  job->next_in_ptr = reinterpret_cast<uint8_t*>(const_cast<char*>(src));
  job->next_out_ptr = reinterpret_cast<uint8_t*>(dest);
  job->available_in = static_cast<uint32_t>(srcLength);
  job->available_out = static_cast<uint32_t>(destLength);
  job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_ZLIB_MODE;

  qpl_status status = qpl_submit_job(job);
  if (status == QPL_STS_QUEUES_ARE_BUSY_ERR) {
    qpl_job_pool.releaseJob(deflate_job.first);
    deflate_job = qpl_job_pool.acquireDeflateJob();
    job = deflate_job.second;
    if (job == nullptr) {
      LOG(WARNING)
          << "cannot acqure deflate job after QPL_STS_QUEUES_ARE_BUSY_ERR ";
      return folly::makeSemiFutureWith([]() -> uint64_t {
        throw std::runtime_error(
            "Cannot acqure deflate job from pool after QPL_STS_QUEUES_ARE_BUSY_ERR");
        return 0;
      });
    }
    job->op = qpl_op_decompress;
    job->next_in_ptr = reinterpret_cast<uint8_t*>(const_cast<char*>(src));
    job->next_out_ptr = reinterpret_cast<uint8_t*>(dest);
    job->available_in = static_cast<uint32_t>(srcLength);
    job->available_out = static_cast<uint32_t>(destLength);
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_ZLIB_MODE;

    status = qpl_submit_job(job);
  }
  if (status != QPL_STS_OK) {
    qpl_job_pool.releaseJob(deflate_job.first);
    LOG(WARNING) << "cannot submit job, error status: " << status;
    return folly::makeSemiFutureWith([this, status]() -> uint64_t {
      throw std::runtime_error("Cannot submit job, error status: " + status);
      return 0;
    });
  } else {
    return folly::makeSemiFuture().deferValue(
        [this, deflate_job](auto&&) -> uint64_t {
          return this->waitResult(deflate_job.first);
        });
  }
}

int GzipIAADecompressor::waitResult(int job_id) {
  dwio::common::QplJobHWPool& qpl_job_pool =
      dwio::common::QplJobHWPool::getInstance();
  VELOX_CHECK_LT(job_id, qpl_job_pool.MAX_JOB_NUMBER);
  qpl_job* job = qpl_job_pool.getJobById(job_id);

  auto status = qpl_wait_job(job);
  qpl_job_pool.releaseJob(job_id);
  if (status == QPL_STS_OK) {
    return 1;
  }
  LOG(WARNING) << "Decompress w/IAA error, status: " << status;
  return 0;
}

std::unique_ptr<AsyncDecompressor> MakeIAAGzipCodec() {
  return std::make_unique<GzipIAADecompressor>();
}

} // namespace facebook::velox::dwio::common::compression