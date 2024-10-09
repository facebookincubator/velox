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
  ++activeConnections;
}

void S3Metrics::incrementStartedUploads() {
  ++startedUploads;
}

void S3Metrics::incrementFailedUploads() {
  ++failedUploads;
}

void S3Metrics::incrementSuccessfulUploads() {
  ++successfulUploads;
}

void S3Metrics::incrementMetadataCalls() {
  ++metadataCalls;
}

void S3Metrics::incrementListStatusCalls() {
  ++listStatusCalls;
}

void S3Metrics::incrementListLocatedStatusCalls() {
  ++listLocatedStatusCalls;
}

void S3Metrics::incrementListObjectsCalls() {
  ++listObjectsCalls;
}

void S3Metrics::incrementOtherReadErrors() {
  ++otherReadErrors;
}

void S3Metrics::incrementAwsAbortedExceptions() {
  ++awsAbortedExceptions;
}

void S3Metrics::incrementSocketExceptions() {
  ++socketExceptions;
}

void S3Metrics::incrementGetObjectErrors() {
  ++getObjectErrors;
}

void S3Metrics::incrementGetMetadataErrors() {
  ++getMetadataErrors;
}

void S3Metrics::incrementGetObjectRetries() {
  ++getObjectRetries;
}

void S3Metrics::incrementGetMetadataRetries() {
  ++getMetadataRetries;
}

void S3Metrics::incrementReadRetries() {
  ++readRetries;
}

void S3Metrics::decrementActiveConnections() {
  --activeConnections;
}

uint64_t S3Metrics::getDeltaStartedUploads() {
  return startedUploads - prevStartedUploads;
}

uint64_t S3Metrics::getDeltaFailedUploads() {
  return failedUploads - prevFailedUploads;
}

uint64_t S3Metrics::getDeltaSuccessfulUploads() {
  return successfulUploads - prevSuccessfulUploads;
}

void S3Metrics::resetDeltas() {
  prevStartedUploads = startedUploads;
  prevFailedUploads = failedUploads;
  prevSuccessfulUploads = successfulUploads;
}

} // namespace facebook::velox::filesystems
