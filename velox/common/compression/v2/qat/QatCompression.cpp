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
#include "velox/common/compression/v2/qat/QatCompression.h"

namespace facebook::velox::common::qat {

#define QZ_INIT_FAIL(rc) (QZ_OK != rc && QZ_DUPLICATE != rc)

#define QZ_SETUP_SESSION_FAIL(rc) \
  (QZ_PARAMS == rc || QZ_NOSW_NO_HW == rc || QZ_NOSW_LOW_MEM == rc)

class QatGZipCodec final : public Codec {
 public:
  QatGZipCodec(int32_t compressionLevel, QzPollingMode_T pollingMode)
      : compressionLevel_(compressionLevel) {
    auto rc = qzInit(&qzSession_, /* sw_backup = */ 1);
    if (QZ_INIT_FAIL(rc)) {
      VLOG(1) << "QAT hardware init failed with error: " << rc;
    } else {
      QzSessionParamsDeflate_T params;
      // Get the default parameters.
      qzGetDefaultsDeflate(&params);
      params.common_params.polling_mode = pollingMode;
      params.common_params.comp_lvl = compressionLevel;
      rc = qzSetupSessionDeflate(&qzSession_, &params);
      if (QZ_SETUP_SESSION_FAIL(rc)) {
        VLOG(1) << "QAT setup session failed with error: " << rc;
      }
    }
  }

  ~QatGZipCodec() {
    (void)qzTeardownSession(&qzSession_);
    (void)qzClose(&qzSession_);
  }

  uint64_t decompress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override {
    auto compressedSize = static_cast<uint32_t>(inputLength);
    auto uncompressedSize = static_cast<uint32_t>(outputLength);
    auto ret = qzDecompress(
        &qzSession_, input, &compressedSize, output, &uncompressedSize);
    if (ret == QZ_OK) {
      return static_cast<uint64_t>(uncompressedSize);
    } else if (ret == QZ_PARAMS) {
      VELOX_FAIL("QAT decompression failure: params is invalid");
    } else if (ret == QZ_FAIL) {
      VELOX_FAIL("QAT decompression failure: Function did not succeed");
    } else {
      VELOX_FAIL("QAT decompression failure with error: {}", ret);
    }
  }

  uint64_t maxCompressedLength(uint64_t inputLen) override {
    return qzMaxCompressedLength(static_cast<size_t>(inputLen), &qzSession_);
  }

  uint64_t compress(
      uint64_t inputLength,
      const uint8_t* input,
      uint64_t outputLength,
      uint8_t* output) override {
    auto uncompressedSize = static_cast<uint32_t>(inputLength);
    auto compressedSize = static_cast<uint32_t>(outputLength);
    auto ret = qzCompress(
        &qzSession_, input, &uncompressedSize, output, &compressedSize, 1);
    if (ret == QZ_OK) {
      return static_cast<uint64_t>(compressedSize);
    } else if (ret == QZ_PARAMS) {
      VELOX_FAIL("QAT compression failure: params is invalid");
    } else if (ret == QZ_FAIL) {
      VELOX_FAIL("QAT compression failure: function did not succeed");
    } else {
      VELOX_FAIL("QAT compression failure with error: {}", ret);
    }
  }

  std::shared_ptr<Compressor> makeCompressor() override {
    VELOX_UNSUPPORTED("Streaming compression unsupported with QAT");
  }

  std::shared_ptr<Decompressor> makeDecompressor() override {
    VELOX_UNSUPPORTED("Streaming decompression unsupported with QAT");
  }

  int32_t compressionLevel() const override {
    return compressionLevel_;
  }

  CompressionKind compressionKind() const override {
    return CompressionKind_GZIP;
  }

  int32_t minimumCompressionLevel() const override {
    return QZ_DEFLATE_COMP_LVL_MINIMUM;
  }
  int32_t maximumCompressionLevel() const override {
    return QZ_DEFLATE_COMP_LVL_MAXIMUM;
  }
  int32_t defaultCompressionLevel() const override {
    return QZ_COMP_LEVEL_DEFAULT;
  }

 private:
  int32_t compressionLevel_;
  QzSession_T qzSession_ = {0};
};

std::unique_ptr<Codec> makeQatGzipCodec(
    int32_t compressionLevel,
    QzPollingMode_T pollingMode) {
  return std::make_unique<QatGZipCodec>(compressionLevel, pollingMode);
}
} // namespace facebook::velox::common::qat