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
#include <algorithm>

#include "velox/common/base/Exceptions.h"
#include "velox/common/compression/v2/iaa/IaaCompression.h"
#include "velox/common/compression/v2/iaa/QplJobPool.h"

namespace facebook::velox::common::iaa {
namespace {
constexpr int64_t kHwCodecError = -1;

void hwFallback(const std::string& source, qpl_status status) {
  VLOG(1) << "DeflateQpl hardware codec failed, "
          << "falling back to software codec. "
          << "(Details: " << source << " failed, status: " << status
          << " - Please refer to qpl_status in "
          << "./contrib/qpl/include/qpl/c_api/status.h)";
}

void swError(const std::string& source, qpl_status status) {
  VELOX_FAIL(
      "Execution of DeflateQpl software fallback codec failed. "
      "(Details: {} failed, status: {}"
      " - Please refer to qpl_status in "
      "./contrib/qpl/include/qpl/c_api/status.h)",
      source,
      status);
}
} // namespace

class HardwareCodecDeflateQpl {
 public:
  // Stands for hardware codec fail, need fallback to software codec.
  explicit HardwareCodecDeflateQpl(qpl_compression_levels compressionLevel)
      : compressionLevel_(compressionLevel){};

  int64_t doCompressData(
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    uint32_t jobId;
    if (auto jobPtr = acquireCompressionJob(
            jobId, inputLength, input, outputLength, output)) {
      auto status = qpl_execute_job(jobPtr);
      if (status == QPL_STS_OK) {
        uint32_t compressedSize = jobPtr->total_out;
        QplJobHWPool::getInstance().releaseJob(jobId);
        return compressedSize;
      }
      hwFallback("doCompressData->qpl_execute_job", status);
      QplJobHWPool::getInstance().releaseJob(jobId);
      return kHwCodecError;
    }
    return kHwCodecError;
  }

  qpl_job* doCompressDataAsync(
      uint32_t& jobId,
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    if (auto jobPtr = acquireCompressionJob(
            jobId, inputLength, input, outputLength, output)) {
      auto status = qpl_submit_job(jobPtr);
      if (status == QPL_STS_OK) {
        return jobPtr;
      }
      QplJobHWPool::getInstance().releaseJob(jobId);
    }
    return nullptr;
  }

  // Submit job request to the IAA hardware and then busy waiting till it
  // complete.
  int64_t doDecompressData(
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    uint32_t jobId = 0;
    if (auto jobPtr = acquireDecompressionJob(
            jobId, inputLength, input, outputLength, output)) {
      auto status = qpl_execute_job(jobPtr);
      if (status == QPL_STS_OK) {
        uint32_t decompressedSize = jobPtr->total_out;
        QplJobHWPool::getInstance().releaseJob(jobId);
        return decompressedSize;
      }
      hwFallback("doDecompressData->qpl_execute_job", status);
      QplJobHWPool::getInstance().releaseJob(jobId);
      return kHwCodecError;
    }
    return kHwCodecError;
  }

  qpl_job* doDecompressDataAsync(
      uint32_t& jobId,
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    if (auto jobPtr = acquireDecompressionJob(
            jobId, inputLength, input, outputLength, output)) {
      auto status = qpl_submit_job(jobPtr);
      if (status == QPL_STS_OK) {
        return jobPtr;
      }
      QplJobHWPool::getInstance().releaseJob(jobId);
    }
    return nullptr;
  }

 private:
  qpl_compression_levels compressionLevel_;

  qpl_job* acquireDecompressionJob(
      uint32_t& jobId,
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    qpl_job* jobPtr;
    if (!(jobPtr = QplJobHWPool::getInstance().acquireJob(jobId))) {
      VLOG(1) << "DeflateQpl HW codec failed, falling back to SW codec. "
              << "(Details: doDecompressData->AcquireJob fail, "
              << "probably job pool exhausted)";
      return nullptr;
    }
    // Setup qpl decompression job.
    jobPtr->op = qpl_op_decompress;
    jobPtr->next_in_ptr = const_cast<uint8_t*>(input);
    jobPtr->next_out_ptr = output;
    jobPtr->available_in = inputLength;
    jobPtr->available_out = outputLength;
    jobPtr->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;
    return jobPtr;
  }

  qpl_job* acquireCompressionJob(
      uint32_t& jobId,
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    qpl_job* jobPtr;
    if (!(jobPtr = QplJobHWPool::getInstance().acquireJob(jobId))) {
      VLOG(1) << "DeflateQpl HW codec failed, falling back to SW codec. "
              << "(Details: doCompressData->AcquireJob fail, "
              << "probably job pool exhausted)";
      return nullptr;
    }
    // Setup qpl compression job.
    jobPtr->op = qpl_op_compress;
    jobPtr->next_in_ptr = const_cast<uint8_t*>(input);
    jobPtr->next_out_ptr = output;
    jobPtr->available_in = inputLength;
    jobPtr->level = compressionLevel_;
    jobPtr->available_out = outputLength;
    jobPtr->flags = QPL_FLAG_FIRST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_LAST |
        QPL_FLAG_OMIT_VERIFY;
    return jobPtr;
  }
};

class SoftwareCodecDeflateQpl final {
 public:
  explicit SoftwareCodecDeflateQpl(qpl_compression_levels compressionLevel)
      : compressionLevel_(compressionLevel){};

  ~SoftwareCodecDeflateQpl() {
    if (swJob_) {
      qpl_fini_job(swJob_);
    }
  }

  int64_t doCompressData(
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    qpl_job* jobPtr = getJobCodecPtr();
    // Performing a compression operation
    jobPtr->op = qpl_op_compress;
    jobPtr->next_in_ptr = const_cast<uint8_t*>(input);
    jobPtr->next_out_ptr = output;
    jobPtr->available_in = inputLength;
    jobPtr->level = compressionLevel_;
    jobPtr->available_out = outputLength;
    jobPtr->flags = QPL_FLAG_FIRST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_LAST |
        QPL_FLAG_OMIT_VERIFY;

    if (auto status = qpl_execute_job(jobPtr); status != QPL_STS_OK) {
      swError("doCompressData->qpl_execute_job", status);
    }
    return jobPtr->total_out;
  }

  int64_t doDecompressData(
      uint32_t inputLength,
      const uint8_t* input,
      uint32_t outputLength,
      uint8_t* output) {
    qpl_job* jobPtr = getJobCodecPtr();

    // Performing a decompression operation
    jobPtr->op = qpl_op_decompress;
    jobPtr->next_in_ptr = const_cast<uint8_t*>(input);
    jobPtr->next_out_ptr = output;
    jobPtr->available_in = inputLength;
    jobPtr->available_out = outputLength;
    jobPtr->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;

    if (auto status = qpl_execute_job(jobPtr); status != QPL_STS_OK) {
      swError("doDecompressData->qpl_execute_job", status);
    }
    return jobPtr->total_out;
  }

 private:
  qpl_job* swJob_{nullptr};
  std::vector<uint8_t> swBuffer_;
  qpl_compression_levels compressionLevel_;

  qpl_job* getJobCodecPtr() {
    if (!swJob_) {
      uint32_t size = 0;
      qpl_get_job_size(qpl_path_software, &size);
      swBuffer_.resize(size);
      swJob_ = reinterpret_cast<qpl_job*>(swBuffer_.data());

      // Job initialization
      if (auto status = qpl_init_job(qpl_path_software, swJob_);
          status != QPL_STS_OK)
        VELOX_FAIL(
            "Initialization of DeflateQpl software fallback codec failed. "
            "(Details: qpl_init_job with error code: {}"
            " - Please refer to qpl_status in "
            "./contrib/qpl/include/qpl/c_api/status.h)",
            status);
    }
    return swJob_;
  }
};

class QplGzipCodec : public virtual Codec {
 public:
  explicit QplGzipCodec(qpl_compression_levels compressionLevel)
      : hwCodec_(std::make_unique<HardwareCodecDeflateQpl>(compressionLevel)),
        swCodec_(std::make_unique<SoftwareCodecDeflateQpl>(compressionLevel)) {}

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override {
    auto ret = kHwCodecError;
    if (QplJobHWPool::getInstance().jobPoolReady()) {
      ret = hwCodec_->doCompressData(
          (uint32_t)inputLength, input, (uint32_t)outputLength, output);
    }
    if (ret == kHwCodecError) {
      return static_cast<uint64_t>(swCodec_->doCompressData(
          (uint32_t)inputLength, input, (uint32_t)outputLength, output));
    }
    return static_cast<uint64_t>(ret);
  }

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override {
    auto ret = kHwCodecError;
    if (QplJobHWPool::getInstance().jobPoolReady()) {
      ret = hwCodec_->doDecompressData(
          (uint32_t)inputLength, input, (uint32_t)outputLength, output);
    }
    if (ret == kHwCodecError) {
      return swCodec_->doDecompressData(
          (uint32_t)inputLength, input, (uint32_t)outputLength, output);
    }
    return ret;
  }

  uint64_t maxCompressedLength(uint64_t inputLength) override {
    // Aligned with ZLIB.
    return (
        (inputLength) + ((inputLength) >> 12) + ((inputLength) >> 14) +
        ((inputLength) >> 25) + 13);
  }

  std::shared_ptr<Compressor> makeCompressor() override {
    VELOX_UNSUPPORTED("Streaming compression unsupported with QAT");
  }

  std::shared_ptr<Decompressor> makeDecompressor() override {
    VELOX_UNSUPPORTED("Streaming decompression unsupported with QAT");
  }

  CompressionKind compressionKind() const override {
    return CompressionKind::CompressionKind_GZIP;
  }

  int32_t minimumCompressionLevel() const override {
    return qpl_level_1;
  }
  int32_t maximumCompressionLevel() const override {
    return qpl_high_level;
  }
  int32_t defaultCompressionLevel() const override {
    return qpl_default_level;
  }

 protected:
  std::unique_ptr<HardwareCodecDeflateQpl> hwCodec_;
  std::unique_ptr<SoftwareCodecDeflateQpl> swCodec_;
};

class QplGzipAsyncCodec : public QplGzipCodec, public AsyncCodec {
 public:
  explicit QplGzipAsyncCodec(qpl_compression_levels compressionLevel)
      : QplGzipCodec(compressionLevel) {}

  folly::SemiFuture<uint64_t> decompressAsync(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override {
    uint32_t jobId;
    if (auto jobPtr = hwCodec_->doDecompressDataAsync(
            jobId, inputLength, input, outputLength, output)) {
      // If `qpl_submit_job` succeeds, add `qpl_wait_job` callback.
      return folly::makeSemiFuture().deferValue([=](auto&&) -> uint64_t {
        auto status = qpl_wait_job(jobPtr);
        if (status == QPL_STS_OK) {
          uint32_t decompressedSize = jobPtr->total_out;
          QplJobHWPool::getInstance().releaseJob(jobId);
          return static_cast<uint64_t>(decompressedSize);
        }
        // If `qpl_wait_job` fails, falling back SW code path.
        QplJobHWPool::getInstance().releaseJob(jobId);
        hwFallback("doDecompressData->qpl_submit_job", status);
        auto swDecompressedLength = swCodec_->doDecompressData(
            (uint32_t)inputLength, input, (uint32_t)outputLength, output);
        return static_cast<uint64_t>(swDecompressedLength);
      });
    }
    // If it fails, return SemiFuture ready with exception.
    return folly::makeSemiFutureWith(
        []() -> uint64_t { throw std::runtime_error("Cannot submit job."); });
  }

  folly::SemiFuture<uint64_t> compressAsync(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) {
    uint32_t jobId;
    if (auto jobPtr = hwCodec_->doCompressDataAsync(
            jobId, inputLength, input, outputLength, output)) {
      // If `qpl_submit_job` succeeds, add `qpl_wait_job` callback.
      return folly::makeSemiFuture().deferValue([=](auto&&) -> uint64_t {
        auto status = qpl_wait_job(jobPtr);
        if (status == QPL_STS_OK) {
          uint32_t compressedSize = jobPtr->total_out;
          QplJobHWPool::getInstance().releaseJob(jobId);
          return static_cast<uint64_t>(compressedSize);
        }
        // If `qpl_wait_job` fails, falling back SW code path.
        QplJobHWPool::getInstance().releaseJob(jobId);
        hwFallback("doDecompressData->qpl_wait_job", status);
        auto swCompressedLength = swCodec_->doCompressData(
            (uint32_t)inputLength, input, (uint32_t)outputLength, output);
        return static_cast<uint64_t>(swCompressedLength);
      });
    }
    // If it fails, return SemiFuture ready with exception.
    return folly::makeSemiFutureWith(
        []() -> uint64_t { throw std::runtime_error("Cannot submit job."); });
  }
};

std::unique_ptr<Codec> makeIaaGzipCodec(
    int32_t compressionLevel,
    uint32_t maxJobNumber) {
  QplJobHWPool::initialize(maxJobNumber);
  return std::make_unique<QplGzipCodec>(
      static_cast<qpl_compression_levels>(compressionLevel));
}

std::unique_ptr<AsyncCodec> makeIaaGzipAsyncCodec(
    int32_t compressionLevel,
    uint32_t maxJobNumber) {
  QplJobHWPool::initialize(maxJobNumber);
  return std::make_unique<QplGzipAsyncCodec>(
      static_cast<qpl_compression_levels>(compressionLevel));
}
} // namespace facebook::velox::common::iaa
