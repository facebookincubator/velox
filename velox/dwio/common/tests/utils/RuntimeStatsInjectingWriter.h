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

#include "velox/dwio/common/WriterFactory.h"

namespace facebook::velox::dwio::common::test {

/// Test-only Writer decorator that delegates all I/O to a wrapped writer
/// and overlays a caller-supplied runtimeStats() map. Used to verify the
/// writer-runtimeStats -> DataSink -> TableWriter -> backgroundTiming path.
class RuntimeStatsInjectingWriter : public Writer {
 public:
  RuntimeStatsInjectingWriter(
      std::unique_ptr<Writer> delegate,
      folly::F14FastMap<std::string, RuntimeMetric> injectedStats)
      : delegate_(std::move(delegate)),
        injectedStats_(std::move(injectedStats)) {}

  // NOTE: state_ is not synced from the delegate. Tests that introspect
  // state() should query the delegate directly.

  void write(const VectorPtr& data) override {
    delegate_->write(data);
  }

  void flush() override {
    delegate_->flush();
  }

  bool finish() override {
    return delegate_->finish();
  }

  std::unique_ptr<dwio::common::FileMetadata> close() override {
    return delegate_->close();
  }

  void abort() override {
    delegate_->abort();
  }

  folly::F14FastMap<std::string, RuntimeMetric> runtimeStats() const override {
    auto stats = delegate_->runtimeStats();
    for (const auto& [name, metric] : injectedStats_) {
      auto [it, inserted] = stats.emplace(name, metric);
      if (!inserted) {
        it->second.merge(metric);
      }
    }
    return stats;
  }

 private:
  std::unique_ptr<Writer> delegate_;
  const folly::F14FastMap<std::string, RuntimeMetric> injectedStats_;
};

class RuntimeStatsInjectingWriterFactory : public WriterFactory {
 public:
  RuntimeStatsInjectingWriterFactory(
      std::shared_ptr<WriterFactory> delegate,
      folly::F14FastMap<std::string, RuntimeMetric> injectedStats)
      : WriterFactory(delegate->fileFormat()),
        delegate_(std::move(delegate)),
        injectedStats_(std::move(injectedStats)) {}

  std::unique_ptr<Writer> createWriter(
      std::unique_ptr<FileSink> sink,
      const std::shared_ptr<WriterOptions>& options) override {
    auto writer = delegate_->createWriter(std::move(sink), options);
    return std::make_unique<RuntimeStatsInjectingWriter>(
        std::move(writer), injectedStats_);
  }

  std::unique_ptr<WriterOptions> createWriterOptions() override {
    return delegate_->createWriterOptions();
  }

 private:
  std::shared_ptr<WriterFactory> delegate_;
  const folly::F14FastMap<std::string, RuntimeMetric> injectedStats_;
};

} // namespace facebook::velox::dwio::common::test
