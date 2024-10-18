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
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::filesystems {

// S3 Metric names
constexpr auto kMetricS3ActiveConnections = "S3ActiveConnections";
constexpr auto kMetricS3MetadataCalls = "S3MetadataCalls";
constexpr auto kMetricS3ListStatusCalls = "S3ListStatusCalls";
constexpr auto kMetricS3ListLocatedStatusCalls = "S3ListLocatedStatusCalls";
constexpr auto kMetricS3ListObjectsCalls = "S3ListObjectsCalls";
constexpr auto kMetricS3OtherReadErrors = "S3OtherReadErrors";
constexpr auto kMetricS3AwsAbortedExceptions = "S3AwsAbortedExceptions";
constexpr auto kMetricS3SocketExceptions = "S3SocketExceptions";
constexpr auto kMetricS3GetObjectErrors = "S3GetObjectErrors";
constexpr auto kMetricS3GetMetadataErrors = "S3GetMetadataErrors";
constexpr auto kMetricS3GetObjectRetries = "S3GetObjectRetries";
constexpr auto kMetricS3GetMetadataRetries = "S3GetMetadataRetries";
constexpr auto kMetricS3ReadRetries = "S3ReadRetries";
constexpr auto kMetricS3StartedUploads = "S3StartedUploads";
constexpr auto kMetricS3FailedUploads = "S3FailedUploads";
constexpr auto kMetricS3SuccessfulUploads = "S3SuccessfulUploads";

// Struct to hold S3-related metrics with delta tracking.
struct S3Metrics : public FileSystemMetrics {
  std::atomic<uint64_t> activeConnections{0};
  std::atomic<uint64_t> startedUploads{0}, prevStartedUploads{0};
  std::atomic<uint64_t> failedUploads{0}, prevFailedUploads{0};
  std::atomic<uint64_t> successfulUploads{0}, prevSuccessfulUploads{0};
  std::atomic<uint64_t> metadataCalls{0};
  std::atomic<uint64_t> listStatusCalls{0};
  std::atomic<uint64_t> listLocatedStatusCalls{0};
  std::atomic<uint64_t> listObjectsCalls{0};
  std::atomic<uint64_t> otherReadErrors{0};
  std::atomic<uint64_t> awsAbortedExceptions{0};
  std::atomic<uint64_t> socketExceptions{0};
  std::atomic<uint64_t> getObjectErrors{0};
  std::atomic<uint64_t> getMetadataErrors{0};
  std::atomic<uint64_t> getObjectRetries{0};
  std::atomic<uint64_t> getMetadataRetries{0};
  std::atomic<uint64_t> readRetries{0};

  // Implement pure virtual methods from FileSystemMetrics
  uint64_t getActiveConnections() const override {
    return activeConnections.load();
  }

  uint64_t getMetadataCalls() const override {
    return metadataCalls.load();
  }

  uint64_t getStartedUploads() const override {
    return startedUploads.load();
  }

  uint64_t getFailedUploads() const override {
    return failedUploads.load();
  }

  uint64_t getSuccessfulUploads() const override {
    return successfulUploads.load();
  }

  // Method to increment each metric based on its name
  void incrementActiveConnections();
  void incrementStartedUploads();
  void incrementFailedUploads();
  void incrementSuccessfulUploads();
  void incrementMetadataCalls();
  void incrementListStatusCalls();
  void incrementListLocatedStatusCalls();
  void incrementListObjectsCalls();
  void incrementOtherReadErrors();
  void incrementAwsAbortedExceptions();
  void incrementSocketExceptions();
  void incrementGetObjectErrors();
  void incrementGetMetadataErrors();
  void incrementGetObjectRetries();
  void incrementGetMetadataRetries();
  void incrementReadRetries();
  void decrementActiveConnections();

  // Get the delta (change) between two consecutive metric updates.
  uint64_t getDeltaStartedUploads();
  uint64_t getDeltaFailedUploads();
  uint64_t getDeltaSuccessfulUploads();

  // Reset the deltas after reporting.
  void resetDeltas();
};

// Global instance of S3Metrics
extern S3Metrics globalS3Metrics;

} // namespace facebook::velox::filesystems
