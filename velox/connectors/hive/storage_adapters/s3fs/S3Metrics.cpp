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

#include "S3Metrics.h"
#include <cstdint>

namespace facebook::velox::filesystems {

// Global instance of S3Metrics
S3Metrics globalS3Metrics;

// Method implementations for S3Metrics
void S3Metrics::incrementActiveConnections() {
  activeConnections.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementStartedUploads() {
  startedUploads.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementFailedUploads() {
  failedUploads.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementSuccessfulUploads() {
  successfulUploads.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementMetadataCalls() {
  metadataCalls.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementListStatusCalls() {
  listStatusCalls.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementListLocatedStatusCalls() {
  listLocatedStatusCalls.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementListObjectsCalls() {
  listObjectsCalls.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementOtherReadErrors() {
  otherReadErrors.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementAwsAbortedExceptions() {
  awsAbortedExceptions.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementSocketExceptions() {
  socketExceptions.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementGetObjectErrors() {
  getObjectErrors.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementGetMetadataErrors() {
  getMetadataErrors.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementGetObjectRetries() {
  getObjectRetries.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementGetMetadataRetries() {
  getMetadataRetries.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::incrementReadRetries() {
  readRetries.fetch_add(1, std::memory_order_relaxed);
}

void S3Metrics::decrementActiveConnections() {
  activeConnections.fetch_sub(1, std::memory_order_relaxed);
}

uint64_t S3Metrics::getDeltaStartedUploads() {
  uint64_t delta = startedUploads.load(std::memory_order_relaxed) - prevStartedUploads;
  prevStartedUploads = startedUploads.load(std::memory_order_relaxed);
  return delta;
}

uint64_t S3Metrics::getDeltaFailedUploads() {
  uint64_t delta = failedUploads.load(std::memory_order_relaxed) - prevFailedUploads;
  prevFailedUploads = failedUploads.load(std::memory_order_relaxed);
  return delta;
}

uint64_t S3Metrics::getDeltaSuccessfulUploads() {
  uint64_t delta = successfulUploads.load(std::memory_order_relaxed) - prevSuccessfulUploads;
  prevSuccessfulUploads = successfulUploads.load(std::memory_order_relaxed);
  return delta;
}

void S3Metrics::resetDeltas() {
  prevStartedUploads = startedUploads.load(std::memory_order_relaxed);
  prevFailedUploads = failedUploads.load(std::memory_order_relaxed);
  prevSuccessfulUploads = successfulUploads.load(std::memory_order_relaxed);
}

} // namespace facebook::velox::filesystems
