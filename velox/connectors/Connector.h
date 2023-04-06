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

#include "velox/common/base/AsyncSource.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/core/Context.h"
#include "velox/vector/ComplexVector.h"

#include <folly/Synchronized.h>

namespace facebook::velox::common {
class Filter;
}

namespace facebook::velox::core {
class ITypedExpr;
} // namespace facebook::velox::core

namespace facebook::velox::exec {
class ExprSet;
}

namespace facebook::velox::connector {

class DataSource;

// A split represents a chunk of data that a connector should load and return
// as a RowVectorPtr, potentially after processing pushdowns.
struct ConnectorSplit {
  const std::string connectorId;

  std::shared_ptr<AsyncSource<std::shared_ptr<DataSource>>> dataSource;

  explicit ConnectorSplit(const std::string& _connectorId)
      : connectorId(_connectorId) {}

  virtual ~ConnectorSplit() {}

  virtual std::string toString() const {
    return fmt::format("[split: {}]", connectorId);
  }
};

class ColumnHandle {
 public:
  virtual ~ColumnHandle() = default;
};

class ConnectorTableHandle {
 public:
  explicit ConnectorTableHandle(std::string connectorId)
      : connectorId_(std::move(connectorId)) {}

  virtual ~ConnectorTableHandle() = default;

  virtual std::string toString() const = 0;

  const std::string& connectorId() const {
    return connectorId_;
  }

 private:
  const std::string connectorId_;
};

/**
 * Represents a request for writing to connector
 */
class ConnectorInsertTableHandle {
 public:
  virtual ~ConnectorInsertTableHandle() {}

  // Whether multi-threaded write is supported by this connector. Planner uses
  // this flag to determine number of drivers.
  virtual bool supportsMultiThreading() const {
    return false;
  }
};

/// Represents the commit strategy for writing to connector.
enum class CommitStrategy {
  kNoCommit, // No more commit actions are needed.
  kTaskCommit // Task level commit is needed.
};

/// Return a string encoding of the given commit strategy.
std::string commitStrategyToString(CommitStrategy commitStrategy);

class DataSink {
 public:
  virtual ~DataSink() = default;

  /// Add the next data (vector) to be written. This call is blocking.
  // TODO maybe at some point we want to make it async.
  virtual void appendData(RowVectorPtr input) = 0;

  /// Called once after all data has been added via possibly multiple calls to
  /// appendData(). Could return data in the string form that would be included
  /// in the output. After calling this function, only close() could be called.
  virtual std::vector<std::string> finish() const = 0;

  virtual void close() = 0;
};

class DataSource {
 public:
  static constexpr int64_t kUnknownRowSize = -1;
  virtual ~DataSource() = default;

  // Add split to process, then call next multiple times to process the split.
  // A split must be fully processed by next before another split can be
  // added. Next returns nullptr to indicate that current split is fully
  // processed.
  virtual void addSplit(std::shared_ptr<ConnectorSplit> split) = 0;

  // Process a split added via addSplit. Returns nullptr if split has been fully
  // processed. Returns std::nullopt and sets the 'future' if started
  // asynchronous work and needs to wait for it to complete to continue
  // processing. The caller will wait for the 'future' to complete before
  // calling 'next' again.
  virtual std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& future) = 0;

  // Add dynamically generated filter.
  // @param outputChannel index into outputType specified in
  // Connector::createDataSource() that identifies the column this filter
  // applies to.
  virtual void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) = 0;

  // Returns the number of input bytes processed so far.
  virtual uint64_t getCompletedBytes() = 0;

  // Returns the number of input rows processed so far.
  virtual uint64_t getCompletedRows() = 0;

  virtual std::unordered_map<std::string, RuntimeCounter> runtimeStats() = 0;

  // Returns true if 'this' has initiated all the prefetch this will
  // initiate. This means that the caller should schedule next splits
  // to prefetch in the background. false if the source does not
  // prefetch.
  virtual bool allPrefetchIssued() const {
    return false;
  }

  // Initializes this from 'source'. 'source' is effectively moved
  // into 'this' Adaptation like dynamic filters stay in effect but
  // the parts dealing with open files, prefetched data etc. are moved. 'source'
  // is freed after the move.
  virtual void setFromDataSource(std::shared_ptr<DataSource> /*source*/) {
    VELOX_UNSUPPORTED("setFromDataSource");
  }

  // Returns a connector dependent row size if available. This can be
  // called after addSplit().  This estimates uncompressed data
  // sizes. This is better than getCompletedBytes()/getCompletedRows()
  // since these track sizes before decompression and may include
  // read-ahead and extra IO from coalescing reads and  will not
  // fully account for size of sparsely accessed columns.
  virtual int64_t estimatedRowSize() {
    return kUnknownRowSize;
  }
};

// Exposes expression evaluation functionality of the engine to the
// connector.  Connector may use it, for example, to evaluate pushed
// down filters. This is not thread safe and serializing operations is
// the responsibility of the caller. This is self-contained and does
// not reference objects from the thread which constructs
// this. Passing this between threads is allowed as long as uses are
// sequential. May reference query-level structures like QueryCtx.
class ExpressionEvaluator {
 public:
  virtual ~ExpressionEvaluator() = default;

  // Compiles an expression. Returns an instance of exec::ExprSet that can be
  // used to evaluate that expression on multiple vectors using evaluate method.
  virtual std::unique_ptr<exec::ExprSet> compile(
      const std::shared_ptr<const core::ITypedExpr>& expression) const = 0;

  // Evaluates previously compiled expression on the specified rows.
  // Re-uses result vector if it is not null.
  virtual void evaluate(
      exec::ExprSet* FOLLY_NONNULL exprSet,
      const SelectivityVector& rows,
      RowVectorPtr& input,
      VectorPtr* FOLLY_NULLABLE result) const = 0;
};

/// Collection of context data for use in a DataSource or DataSink. One instance
/// of this per DataSource and DataSink. This may be passed between threads but
/// methods must be invoked sequentially. Serializing use is the responsibility
/// of the caller.
class ConnectorQueryCtx {
 public:
  ConnectorQueryCtx(
      memory::MemoryPool* leafPool,
      memory::MemoryPool* aggregatePool,
      const Config* connectorConfig,
      std::unique_ptr<ExpressionEvaluator> expressionEvaluator,
      memory::MemoryAllocator* FOLLY_NONNULL allocator,
      const std::string& taskId,
      const std::string& planNodeId,
      int driverId)
      : leafPool_(leafPool),
        aggregatePool_(aggregatePool),
        config_(connectorConfig),
        expressionEvaluator_(std::move(expressionEvaluator)),
        allocator_(allocator),
        scanId_(fmt::format("{}.{}", taskId, planNodeId)),
        taskId_(taskId),
        driverId_(driverId) {
    VELOX_CHECK_NOT_NULL(leafPool_);
  }

  /// Returns the memory pool for memory allocation.
  memory::MemoryPool* memoryPool() const {
    return leafPool_;
  }

  /// Returns the aggregate memory pool for the data sink that needs the
  /// hierarchical memory pool management, such as HiveDataSink. This is set to
  /// null for table scan.
  memory::MemoryPool* aggregatePool() const {
    VELOX_CHECK_NOT_NULL(aggregatePool_);
    return aggregatePool_;
  }

  const Config* FOLLY_NONNULL config() const {
    return config_;
  }

  ExpressionEvaluator* FOLLY_NULLABLE expressionEvaluator() const {
    return expressionEvaluator_.get();
  }

  // MemoryAllocator for large allocations. Used for caching with
  // CachedBufferedImput if this implements cache::AsyncDataCache.
  memory::MemoryAllocator* FOLLY_NONNULL allocator() const {
    return allocator_;
  }

  // This is a combination of task id and the scan's PlanNodeId. This is an id
  // that allows sharing state between different threads of the same scan. This
  // is used for locating a scanTracker, which tracks the read density of
  // columns for prefetch and other memory hierarchy purposes.
  const std::string& scanId() const {
    return scanId_;
  }

  const std::string& taskId() const {
    return taskId_;
  }

  int driverId() const {
    return driverId_;
  }

 private:
  memory::MemoryPool* leafPool_;
  memory::MemoryPool* aggregatePool_;
  const Config* FOLLY_NONNULL config_;
  std::unique_ptr<ExpressionEvaluator> expressionEvaluator_;
  memory::MemoryAllocator* FOLLY_NONNULL allocator_;
  const std::string scanId_;
  const std::string taskId_;
  const int driverId_;
};

class Connector {
 public:
  explicit Connector(
      const std::string& id,
      std::shared_ptr<const Config> properties)
      : id_(id), properties_(std::move(properties)) {}

  virtual ~Connector() = default;

  const std::string& connectorId() const {
    return id_;
  }

  const std::shared_ptr<const Config>& connectorProperties() const {
    return properties_;
  }

  // Returns true if this connector would accept a filter dynamically generated
  // during query execution.
  virtual bool canAddDynamicFilter() const {
    return false;
  }

  virtual std::shared_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      ConnectorQueryCtx* FOLLY_NONNULL connectorQueryCtx) = 0;

  // Returns true if addSplit of DataSource can use 'dataSource' from
  // ConnectorSplit in addSplit(). If so, TableScan can preload splits
  // so that file opening and metadata operations are off the Driver'
  // thread.
  virtual bool supportsSplitPreload() {
    return false;
  }

  virtual std::shared_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<ConnectorInsertTableHandle> connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) = 0;

  // Returns a ScanTracker for 'id'. 'id' uniquely identifies the
  // tracker and different threads will share the same
  // instance. 'loadQuantum' is the largest single IO for the query
  // being tracked.
  static std::shared_ptr<cache::ScanTracker> getTracker(
      const std::string& scanId,
      int32_t loadQuantum);

  virtual folly::Executor* FOLLY_NULLABLE executor() const {
    return nullptr;
  }

 private:
  static void unregisterTracker(cache::ScanTracker* FOLLY_NONNULL tracker);

  const std::string id_;

  static folly::Synchronized<
      std::unordered_map<std::string_view, std::weak_ptr<cache::ScanTracker>>>
      trackers_;

  const std::shared_ptr<const Config> properties_;
};

class ConnectorFactory {
 public:
  explicit ConnectorFactory(const char* FOLLY_NONNULL name) : name_(name) {}

  virtual ~ConnectorFactory() = default;

  const std::string& connectorName() const {
    return name_;
  }

  virtual std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const Config> properties,
      folly::Executor* FOLLY_NULLABLE executor = nullptr) = 0;

 private:
  const std::string name_;
};

/// Adds a factory for creating connectors to the registry using connector name
/// as the key. Throws if factor with the same name is already present. Always
/// returns true. The return value makes it easy to use with
/// FB_ANONYMOUS_VARIABLE.
bool registerConnectorFactory(std::shared_ptr<ConnectorFactory> factory);

/// Returns a factory for creating connectors with the specified name. Throws if
/// factory doesn't exist.
std::shared_ptr<ConnectorFactory> getConnectorFactory(
    const std::string& connectorName);

/// Adds connector instance to the registry using connector ID as the key.
/// Throws if connector with the same ID is already present. Always returns
/// true. The return value makes it easy to use with FB_ANONYMOUS_VARIABLE.
bool registerConnector(std::shared_ptr<Connector> connector);

/// Removes the connector with specified ID from the registry. Returns true if
/// connector was removed and false if connector didn't exist.
bool unregisterConnector(const std::string& connectorId);

/// Returns a connector with specified ID. Throws if connector doesn't exist.
std::shared_ptr<Connector> getConnector(const std::string& connectorId);

/// Returns a map of all (connectorId -> connector) pairs currently registered.
const std::unordered_map<std::string, std::shared_ptr<Connector>>&
getAllConnectors();

#define VELOX_REGISTER_CONNECTOR_FACTORY(theFactory)                      \
  namespace {                                                             \
  static bool FB_ANONYMOUS_VARIABLE(g_ConnectorFactory) =                 \
      facebook::velox::connector::registerConnectorFactory((theFactory)); \
  }
} // namespace facebook::velox::connector
