/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <folly/json/dynamic.h>

namespace facebook::nimble {

struct StripeLoadMetrics {
  uint32_t stripeIndex;
  uint32_t rowsInStripe;
  uint32_t streamCount{0};
  uint32_t totalStreamSize{0};
  // TODO: add IO sizes.

  // TODO: add encoding summary

  size_t cpuUsec;
  size_t wallTimeUsec;

  folly::dynamic serialize() const;
};

// Might be good to capture via some kind of run stats struct.
// We can then adapt the run stats to file writer run stats.
struct StripeFlushMetrics {
  // Stripe shape summary.
  uint64_t inputSize;
  uint64_t rowCount;
  uint64_t stripeSize;

  // We would add flush policy states here when wired up in the future.
  // uint64_t inputChunkSize_;

  // TODO: add some type of encoding summary in a follow-up diff.

  // Memory footprint
  uint64_t trackedMemory;
  // uint64_t residentMemory;

  // Perf stats.
  uint64_t flushCpuUsec;
  uint64_t flushWallTimeUsec;
  // Add IOStatistics when we have finished WS api consolidations.

  folly::dynamic serialize() const;
};

struct FileCloseMetrics {
  uint64_t rowCount;
  uint64_t inputSize;
  uint64_t stripeCount;
  uint64_t fileSize;

  // Perf stats.
  uint64_t encodingCpuNs;
  uint64_t encodingWallNs;
  // Add IOStatistics when we have finished WS api consolidations.

  folly::dynamic serialize() const;
};

enum class LogOperation {
  Write,
  Flush,
  Close,
  StripeLoad,
  CompressionContext,
};

class MetricsLogger {
 public:
  virtual ~MetricsLogger() = default;

  virtual void logException(
      LogOperation /* operation */,
      const std::string& /* errorMessage */) const {}

  virtual void logStripeLoad(const StripeLoadMetrics& /* metrics */) const {}
  virtual void logStripeFlush(const StripeFlushMetrics& /* metrics */) const {}
  virtual void logFileClose(const FileCloseMetrics& /* metrics */) const {}
  virtual void logCompressionContext(const std::string&) const {}
};

class LoggingScope {
 public:
  explicit LoggingScope(const MetricsLogger& logger) {
    Context::get().logger = &logger;
  }

  ~LoggingScope() {
    Context::get().logger = nullptr;
  }

  static const MetricsLogger* getLogger() {
    return Context::get().logger;
  }

 private:
  struct Context {
    const MetricsLogger* logger;

    static Context& get();
  };
};

} // namespace facebook::nimble
