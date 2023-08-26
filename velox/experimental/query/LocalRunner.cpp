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

#include "velox/experimental/query/LocalRunner.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

namespace facebook::velox::exec {
namespace {
auto remoteSplit(const std::string& taskId) {
  return std::make_shared<RemoteConnectorSplit>(taskId);
}
} // namespace

test::TaskCursor* LocalRunner::cursor() {
  auto lastStage = makeStages();
  params_.planNode = plan_.back().fragment.planNode;
  auto cursor = std::make_unique<test::TaskCursor>(params_);
  stages_.push_back({cursor->task()});
  if (!lastStage.empty()) {
    auto node = plan_.back().inputStages[0].consumer;
    for (auto& remote : lastStage) {
      cursor->task()->addSplit(node, Split(remote));
    }
    cursor->task()->noMoreSplits(node);
  }
  {
    std::lock_guard<std::mutex> l(mutex_);
    tasksCreated_ = true;
    if (!error_) {
      cursor_ = std::move(cursor);
    }
  }
  if (!cursor_) {
    // The cursor was not set because previous fragments had an error.
    terminate();
    std::rethrow_exception(error_);
  }
  return cursor_.get();
}

void LocalRunner::terminate() {
  VELOX_CHECK(tasksCreated_);
  for (auto& stage : stages_) {
    for (auto& task : stage) {
      task->setError(error_);
    }
  }
  if (cursor_) {
    cursor_->setError(error_);
  }
}

std::vector<std::shared_ptr<RemoteConnectorSplit>> LocalRunner::makeStages() {
  std::unordered_map<std::string, int32_t> prefixMap;
  auto sharedRunner = shared_from_this();
  auto onError = [self = sharedRunner, this](std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (error_) {
        return;
      }
      error_ = e;
    }
    if (cursor_) {
      terminate();
    }
  };

  for (auto fragmentIndex = 0; fragmentIndex < plan_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = plan_[fragmentIndex];
    prefixMap[fragment.taskPrefix] = stages_.size();
    stages_.emplace_back();
    for (auto i = 0; i < fragment.width; ++i) {
      Consumer consumer = nullptr;
      auto task = Task::create(
          fmt::format(
              "local://{}/{}.{}",
              params_.queryCtx->queryId(),
              fragment.taskPrefix,
              i),
          fragment.fragment,
          i,
          params_.queryCtx,
          consumer,
          onError);
      stages_.back().push_back(task);
      if (fragment.numBroadcastDestinations) {
        task->updateOutputBuffers(fragment.numBroadcastDestinations, true);
      }
      Task::start(task, options_.numDrivers);
    }
  }

  for (auto fragmentIndex = 0; fragmentIndex < plan_.size() - 1;
       ++fragmentIndex) {
    auto& fragment = plan_[fragmentIndex];
    for (auto& scan : fragment.scans) {
      auto source = splitSourceFactory_->splitSourceForScan(*scan);
      bool allDone = false;
      do {
        for (auto i = 0; i < stages_[fragmentIndex].size(); ++i) {
          auto split = source->next(i);
          if (!split.hasConnectorSplit()) {
            allDone = true;
            break;
          }
          stages_[fragmentIndex][i]->addSplit(scan->id(), std::move(split));
        }
      } while (!allDone);
    }
    for (auto& scan : fragment.scans) {
      for (auto i = 0; i < stages_[fragmentIndex].size(); ++i) {
        stages_[fragmentIndex][i]->noMoreSplits(scan->id());
      }
    }

    for (auto& input : fragment.inputStages) {
      auto sourceStage = prefixMap[input.producerTaskPrefix];
      std::vector<std::shared_ptr<RemoteConnectorSplit>> sourceSplits;
      for (auto i = 0; i < stages_[sourceStage].size(); ++i) {
        sourceSplits.push_back(remoteSplit(stages_[sourceStage][i]->taskId()));
      }
      for (auto& task : stages_[fragmentIndex]) {
        for (auto& remote : sourceSplits) {
          task->addSplit(input.consumer, Split(remote));
        }
        task->noMoreSplits(input.consumer);
      }
    }
  }
  if (stages_.empty()) {
    return {};
  }
  std::vector<std::shared_ptr<RemoteConnectorSplit>> lastStage;
  for (auto& task : stages_.back()) {
    lastStage.push_back(remoteSplit(task->taskId()));
  }
  return lastStage;
}

Split LocalSplitSource::next(int32_t /*worker*/) {
  if (currentFile_ >= static_cast<int32_t>(table_->files.size())) {
    return Split();
  }
  if (currentSplit_ >= fileSplits_.size()) {
    fileSplits_.clear();
    ++currentFile_;
    if (currentFile_ >= table_->files.size()) {
      return Split();
    }
    currentSplit_ = 0;
    auto filePath = table_->files[currentFile_];
    const int fileSize = fs::file_size(filePath);
    // Take the upper bound.
    const int splitSize = std::ceil((fileSize) / splitsPerFile_);
    for (int i = 0; i < splitsPerFile_; i++) {
      fileSplits_.push_back(test::HiveConnectorSplitBuilder(filePath)
                                .fileFormat(table_->format)
                                .start(i * splitSize)
                                .length(splitSize)
                                .build());
    }
  }
  return Split(std::move(fileSplits_[currentSplit_++]));
}

std::unique_ptr<SplitSource> LocalSplitSourceFactory::splitSourceForScan(
    const core::TableScanNode& tableScan) {
  auto tableHandle = dynamic_cast<const connector::hive::HiveTableHandle*>(
      tableScan.tableHandle().get());
  VELOX_CHECK(tableHandle);
  auto it = schema_.tables().find(tableHandle->tableName());
  VELOX_CHECK(it != schema_.tables().end());
  auto table = it->second.get();
  return std::make_unique<LocalSplitSource>(table, splitsPerFile_);
}

std::vector<TaskStats> LocalRunner::stats() const {
  std::vector<TaskStats> result;
  for (auto i = 0; i < stages_.size(); ++i) {
    auto& tasks = stages_[i];
    assert(!tasks.empty());
    auto stats = tasks[0]->taskStats();
    for (auto j = 1; j < tasks.size(); ++j) {
      auto moreStats = tasks[j]->taskStats();
      for (auto pipeline = 0; pipeline < stats.pipelineStats.size();
           ++pipeline) {
        for (auto op = 0;
             op < stats.pipelineStats[pipeline].operatorStats.size();
             ++op) {
          stats.pipelineStats[pipeline].operatorStats[op].add(
              moreStats.pipelineStats[pipeline].operatorStats[op]);
        }
      }
    }
    result.push_back(std::move(stats));
  }
  return result;
}

} // namespace facebook::velox::exec
