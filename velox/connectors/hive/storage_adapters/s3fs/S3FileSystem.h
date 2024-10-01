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
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Metrics.h"

namespace facebook::velox::filesystems {
using namespace facebook::velox::connector::hive;

bool initializeS3(std::string_view logLevel = "FATAL");
// Struct to hold S3-related metrics with delta tracking.
struct S3Metrics {
  uint64_t activeConnections{0};

  uint64_t startedUploads{0}, prevStartedUploads{0};
  uint64_t failedUploads{0}, prevFailedUploads{0};
  uint64_t successfulUploads{0}, prevSuccessfulUploads{0};

  uint64_t metadataCalls{0};
  uint64_t listStatusCalls{0};
  uint64_t listLocatedStatusCalls{0};
  uint64_t listObjectsCalls{0};
  uint64_t otherReadErrors{0};
  uint64_t awsAbortedExceptions{0};
  uint64_t socketExceptions{0};
  uint64_t getObjectErrors{0};
  uint64_t getMetadataErrors{0};
  uint64_t getObjectRetries{0};
  uint64_t getMetadataRetries{0};
  uint64_t readRetries{0};

  void increment(const std::string& metricName) {
    if (metricName == "activeConnections") {
      ++activeConnections;
    } else if (metricName == "startedUploads") {
      ++startedUploads;
    } else if (metricName == "failedUploads") {
      ++failedUploads;
    } else if (metricName == "successfulUploads") {
      ++successfulUploads;
    } else if (metricName == "metadataCalls") {
      ++metadataCalls;
    } else if (metricName == "listStatusCalls") {
      ++listStatusCalls;
    } else if (metricName == "listLocatedStatusCalls") {
      ++listLocatedStatusCalls;
    } else if (metricName == "listObjectsCalls") {
      ++listObjectsCalls;
    } else if (metricName == "otherReadErrors") {
      ++otherReadErrors;
    } else if (metricName == "awsAbortedExceptions") {
      ++awsAbortedExceptions;
    } else if (metricName == "socketExceptions") {
      ++socketExceptions;
    } else if (metricName == "getObjectErrors") {
      ++getObjectErrors;
    } else if (metricName == "getMetadataErrors") {
      ++getMetadataErrors;
    } else if (metricName == "getObjectRetries") {
      ++getObjectRetries;
    } else if (metricName == "getMetadataRetries") {
      ++getMetadataRetries;
    } else if (metricName == "readRetries") {
      ++readRetries;
    }
  }

  uint64_t getDelta(const std::string& metricName) {
    if (metricName == "startedUploads") {
      return startedUploads - prevStartedUploads;
    } else if (metricName == "failedUploads") {
      return failedUploads - prevFailedUploads;
    } else if (metricName == "successfulUploads") {
      return successfulUploads - prevSuccessfulUploads;
    }
    return 0; // COUNT metrics do not track deltas.
  }

  void resetDeltas() {
    prevStartedUploads = startedUploads;
    prevFailedUploads = failedUploads;
    prevSuccessfulUploads = successfulUploads;
  }
};

// Global instance of S3Metrics
extern S3Metrics globalS3Metrics;

bool initializeS3(const config::ConfigBase* config);

void finalizeS3();

/// Implementation of S3 filesystem and file interface.
/// We provide a registration method for read and write files so the appropriate
/// type of file can be constructed based on a filename.
class S3FileSystem : public FileSystem {
 public:
  S3FileSystem(
      std::string_view bucketName,
      const std::shared_ptr<const config::ConfigBase> config);

  std::string name() const override;

  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view s3Path,
      const FileOptions& options = {}) override;

  std::unique_ptr<WriteFile> openFileForWrite(
      std::string_view s3Path,
      const FileOptions& options) override;

  void remove(std::string_view path) override {
    VELOX_UNSUPPORTED("remove for S3 not implemented");
  }

  void rename(
      std::string_view path,
      std::string_view newPath,
      bool overWrite = false) override {
    VELOX_UNSUPPORTED("rename for S3 not implemented");
  }

  bool exists(std::string_view path) override {
    VELOX_UNSUPPORTED("exists for S3 not implemented");
  }

  std::vector<std::string> list(std::string_view path) override {
    VELOX_UNSUPPORTED("list for S3 not implemented");
  }

  void mkdir(std::string_view path, const DirectoryOptions& options = {})
      override {
    VELOX_UNSUPPORTED("mkdir for S3 not implemented");
  }

  void rmdir(std::string_view path) override {
    VELOX_UNSUPPORTED("rmdir for S3 not implemented");
  }

  std::string getLogLevelName() const;
  
  /// Returns the global S3 metrics.
  const FileSystemMetrics& metrics() const override;

  /// Reset metrics deltas after reporting.
  void resetMetricsDeltas();

  S3Metrics& getMetrics();  // Expose the global metrics
  void resetMetricsDeltas();  // Reset deltas for SUM metrics

 protected:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace facebook::velox::filesystems
