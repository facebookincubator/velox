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
#include "velox/exec/PythonCall.h"

#include "velox/expression/ExprConstants.h"

#include <fmt/ranges.h>
#include <folly/container/F14Map.h>
#include <numeric>

#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace {

// The name of a constant column sent to the worker, followed by its index.
constexpr std::string_view kConstantColumnPrefix{"__constant"};

bool containsMap(const Type& type) {
  if (type.isMap()) {
    return true;
  }
  for (size_t i = 0; i < type.size(); ++i) {
    if (containsMap(*type.childAt(i))) {
      return true;
    }
  }
  return false;
}

// Returns, per element of 'vector', the row of 'results' it belongs to, or -1
// for elements of NULL rows and of rows whose result is NULL, which need no
// more checks.
std::vector<vector_size_t> elementTopRows(
    const ArrayVectorBase& vector,
    const std::vector<vector_size_t>& topRows,
    const BaseVector& results,
    vector_size_t numElements) {
  std::vector<vector_size_t> elementRows(numElements, -1);
  for (vector_size_t row = 0; row < vector.size(); ++row) {
    if (topRows[row] < 0 || vector.isNullAt(row) ||
        results.isNullAt(topRows[row])) {
      continue;
    }
    const auto offset = vector.offsetAt(row);
    for (auto i = offset; i < offset + vector.sizeAt(row); ++i) {
      elementRows[i] = topRows[row];
    }
  }
  return elementRows;
}

// Fails the query, or under TRY sets to NULL, the rows of 'results' whose
// maps, at any depth, repeat a key under Velox's equality. Python can return
// keys that differ in Python but are equal once stored, such as REAL keys.
// 'vector' is part of 'results': its row i belongs to row 'topRows[i]' of
// 'results', or to none if -1. 'results' are the worker's results, copies in
// the pool: flat at every level, which the walk relies on, and changeable in
// place.
void rejectDuplicateMapKeys(
    const BaseVector& vector,
    const std::vector<vector_size_t>& topRows,
    BaseVector& results,
    bool nullOnError,
    std::string_view functionName) {
  const auto recurse = [&](const BaseVector& child,
                           const std::vector<vector_size_t>& childTopRows) {
    if (containsMap(*child.type())) {
      rejectDuplicateMapKeys(
          child, childTopRows, results, nullOnError, functionName);
    }
  };
  switch (vector.typeKind()) {
    case TypeKind::ROW: {
      const auto* row = vector.as<RowVector>();
      VELOX_CHECK_NOT_NULL(row, "Python results must be flat at every level");
      // A NULL row's fields hold no values to check.
      std::vector<vector_size_t> fieldTopRows(topRows);
      for (vector_size_t i = 0; i < row->size(); ++i) {
        if (row->isNullAt(i)) {
          fieldTopRows[i] = -1;
        }
      }
      for (const auto& child : row->children()) {
        recurse(*child, fieldTopRows);
      }
      return;
    }
    case TypeKind::ARRAY: {
      const auto* arrayVector = vector.as<ArrayVector>();
      VELOX_CHECK_NOT_NULL(
          arrayVector, "Python results must be flat at every level");
      const auto& array = *arrayVector;
      recurse(
          *array.elements(),
          elementTopRows(array, topRows, results, array.elements()->size()));
      return;
    }
    case TypeKind::MAP: {
      const auto* mapVector = vector.as<MapVector>();
      VELOX_CHECK_NOT_NULL(
          mapVector, "Python results must be flat at every level");
      const auto& map = *mapVector;
      for (vector_size_t row = 0; row < map.size(); ++row) {
        // A row whose call failed is NULL already.
        if (topRows[row] < 0 || map.isNullAt(row) ||
            results.isNullAt(topRows[row]) ||
            !map.mapKeys()
                 ->findDuplicateValue(
                     map.offsetAt(row), map.sizeAt(row), CompareFlags{})
                 .has_value()) {
          continue;
        }
        VELOX_USER_CHECK(
            nullOnError,
            "Python function must not return a map with duplicate keys: {}",
            functionName);
        results.setNull(topRows[row], true);
      }
      const auto entryTopRows =
          elementTopRows(map, topRows, results, map.mapKeys()->size());
      recurse(*map.mapKeys(), entryTopRows);
      recurse(*map.mapValues(), entryTopRows);
      return;
    }
    default:
      return;
  }
}

} // namespace

PythonCall::PythonCall(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::PythonCallNodePtr& planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          OperatorType::kPythonCall),
      maxPending_{driverCtx->queryConfig().pythonCallMaxPendingBatches()} {
  VELOX_USER_CHECK_GE(
      maxPending_, 1, "python_call_max_pending_batches must be at least 1");
  const auto& sourceType = *planNode->source()->outputType();
  auto& functions = workerOptions_.functions;
  // Equal calls run once in the worker: calls of the same functions on the
  // same arguments, when every function among them is deterministic.
  folly::F14FastMap<std::string, size_t> functionIndices;
  folly::F14FastMap<column_index_t, column_index_t> argumentColumns;

  // Adds the function a call runs, after the functions it reads. Returns its
  // index and whether the call is deterministic, nested calls included.
  std::function<std::pair<size_t, bool>(const core::CallTypedExpr&)>
      addFunction = [&](const core::CallTypedExpr& call) {
        const auto& planFunction = planNode->functions().at(call.name());
        python::PythonFunction function{
            .name = planFunction.name,
            .pickledFunction = planFunction.pickledFunction,
            .resultType = call.type(),
            .output = false,
        };
        bool deterministic = planFunction.deterministic;
        std::vector<std::string> argumentKeys;
        for (const auto& input : call.inputs()) {
          if (input->isCallKind()) {
            const auto [index, nestedDeterministic] =
                addFunction(*input->asUnchecked<core::CallTypedExpr>());
            deterministic = deterministic && nestedDeterministic;
            function.arguments.push_back({.function = index});
            argumentKeys.push_back(fmt::format("f{}", index));
            continue;
          }
          const auto column = [&]() -> column_index_t {
            if (input->isConstantKind()) {
              auto constant =
                  std::static_pointer_cast<const core::ConstantTypedExpr>(
                      input);
              const auto it = std::ranges::find_if(
                  workerColumns_, [&](const WorkerColumn& existing) {
                    return existing.constant != nullptr &&
                        *existing.constant == *constant;
                  });
              if (it != workerColumns_.end()) {
                return static_cast<column_index_t>(it - workerColumns_.begin());
              }
              workerColumns_.push_back({.constant = std::move(constant)});
              return workerColumns_.size() - 1;
            }
            const auto channel = sourceType.getChildIdx(
                input->asUnchecked<core::FieldAccessTypedExpr>()->name());
            const auto [it, inserted] =
                argumentColumns.emplace(channel, workerColumns_.size());
            if (inserted) {
              workerColumns_.push_back({.channel = channel});
            }
            return it->second;
          }();
          function.arguments.push_back({.column = column});
          argumentKeys.push_back(fmt::format("c{}", column));
        }
        if (!deterministic) {
          functions.push_back(std::move(function));
          return std::make_pair(functions.size() - 1, false);
        }
        const auto key =
            fmt::format("{}({})", call.name(), fmt::join(argumentKeys, ","));
        const auto [it, inserted] =
            functionIndices.emplace(key, functions.size());
        if (inserted) {
          functions.push_back(std::move(function));
        }
        return std::make_pair(it->second, true);
      };

  // By output column name: the call's Python call and its function's index.
  folly::F14FastMap<std::string, std::pair<const core::CallTypedExpr*, size_t>>
      callFunctions;
  for (size_t i = 0; i < planNode->calls().size(); ++i) {
    const auto& call = *planNode->calls()[i];
    const auto& pythonCall = call.name() == expression::kTry
        ? *call.inputs()[0]->asUnchecked<core::CallTypedExpr>()
        : call;
    const auto index = addFunction(pythonCall).first;
    functions[index].output = true;
    callFunctions.emplace(planNode->names()[i], std::make_pair(&call, index));
  }

  // RESULT carries the output functions' columns in function order.
  std::vector<size_t> resultIndices(functions.size());
  size_t numOutputs{0};
  for (size_t i = 0; i < functions.size(); ++i) {
    if (functions[i].output) {
      resultIndices[i] = numOutputs++;
    }
  }

  for (column_index_t i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    const auto function = callFunctions.find(name);
    if (function == callFunctions.end()) {
      identityProjections_.emplace_back(sourceType.getChildIdx(name), i);
      continue;
    }
    const auto& [call, index] = function->second;
    resultProjections_.emplace_back(callColumns_.size(), i);
    callColumns_.push_back({
        .resultIndex = resultIndices[index],
        .functionName = functions[index].name,
        .nullOnError = call->name() == expression::kTry,
        .containsMap = containsMap(*functions[index].resultType),
    });
  }
  results_.resize(callColumns_.size());

  std::vector<std::string> argumentNames;
  std::vector<TypePtr> argumentTypes;
  for (const auto& column : workerColumns_) {
    if (column.constant != nullptr) {
      // The worker reads columns by position, so the name is arbitrary.
      argumentNames.push_back(
          fmt::format("{}{}", kConstantColumnPrefix, argumentNames.size()));
      argumentTypes.push_back(column.constant->type());
    } else {
      argumentNames.push_back(sourceType.nameOf(column.channel));
      argumentTypes.push_back(sourceType.childAt(column.channel));
    }
  }
  workerInputType_ = ROW(std::move(argumentNames), std::move(argumentTypes));

  workerOptions_.environmentUri = planNode->environmentUri();
  workerOptions_.resolvers =
      driverCtx->task->queryCtx()->registry<python::PythonEnvironmentResolvers>(
          python::PythonEnvironmentResolvers::kRegistryKey);
  VELOX_CHECK_NOT_NULL(
      workerOptions_.resolvers, "QueryCtx has no Python environment resolvers");
  workerOptions_.pythonVersion = planNode->pythonVersion();
  // Timestamps are converted only when the session adjusts them to a time
  // zone it names, as elsewhere in Velox.
  const auto& queryConfig = driverCtx->queryConfig();
  if (queryConfig.adjustTimestampToTimezone() &&
      !queryConfig.sessionTimezone().empty()) {
    workerOptions_.sessionTimeZone = queryConfig.sessionTimezone();
  }
}

void PythonCall::initialize() {
  Operator::initialize();
  // A complex constant's value is copied flat once here, rather than for
  // every batch the worker client sends.
  for (auto& column : workerColumns_) {
    if (column.constant == nullptr) {
      continue;
    }
    auto value = column.constant->toConstantVector(pool());
    if (!value->type()->isPrimitiveType()) {
      value =
          BaseVector::wrapInConstant(1, 0, BaseVector::copy(*value, pool()));
    }
    column.constantValue = std::move(value);
  }
  addRuntimeStat(
      PythonCallStatNames::kNumFunctions,
      RuntimeCounter(workerOptions_.functions.size()));
  // Holds only reservations, so they add to what the operator allocates.
  workerPool_ = pool()->parent()->addLeafChild(
      fmt::format("{}.pythonWorker", pool()->name()),
      /*threadSafe=*/true,
      exec::MemoryReclaimer::create());
}

bool PythonCall::needsInput() const {
  return !noMoreInput_ && pending_.size() < maxPending_;
}

void PythonCall::addInput(RowVectorPtr input) {
  // A driver that gets no input starts no worker.
  if (worker_ == nullptr) {
    worker_ = std::make_unique<python::PythonWorker>(
        std::move(workerOptions_), pool());
  }
  // Only the columns the calls read go to the worker, and their constants.
  std::vector<VectorPtr> arguments;
  arguments.reserve(workerColumns_.size());
  for (const auto& column : workerColumns_) {
    arguments.push_back(
        column.constant != nullptr
            ? BaseVector::wrapInConstant(input->size(), 0, column.constantValue)
            : input->childAt(column.channel));
  }
  worker_->send(
      std::make_shared<RowVector>(
          pool(),
          workerInputType_,
          nullptr,
          input->size(),
          std::move(arguments)));
  pending_.push_back(std::move(input));
  accountForWorker();
}

void PythonCall::noMoreInput() {
  Operator::noMoreInput();
  if (worker_ != nullptr) {
    worker_->finish();
  }
}

void PythonCall::tryReceive(ContinueFuture* future) {
  if (workerResults_ != nullptr || pending_.empty()) {
    return;
  }
  auto result = worker_->receive(future);
  if (!result.has_value()) {
    return;
  }
  workerResults_ = std::move(result->columns);
  addRuntimeStat(
      PythonCallStatNames::kFunctionWallNanos,
      RuntimeCounter(result->functionWallNanos, RuntimeCounter::Unit::kNanos));
  accountForWorker();
}

void PythonCall::accountForWorker() {
  if (!unpickleRecorded_) {
    if (const auto nanos = worker_->unpickleNanos()) {
      addRuntimeStat(
          PythonCallStatNames::kUnpickleWallNanos,
          RuntimeCounter(nanos.value(), RuntimeCounter::Unit::kNanos));
      unpickleRecorded_ = true;
    }
  }
  // The worker's memory and the batches and results buffered for it, which
  // live outside any pool.
  const auto workerBytes =
      worker_->processStats().peakMemoryBytes + worker_->bufferedBytes();
  if (workerBytes <= reservedWorkerBytes_) {
    return;
  }
  // maybeReserve makes 'workerBytes' of the reservation unused; the pool never
  // allocates, so that is the total reserved.
  if (!workerPool_->maybeReserve(workerBytes)) {
    VELOX_MEM_POOL_CAP_EXCEEDED(
        fmt::format(
            "Python worker memory exceeds what the query may use: {}",
            succinctBytes(workerBytes)));
  }
  reservedWorkerBytes_ = workerBytes;
}

RowVectorPtr PythonCall::getOutput() {
  tryReceive(nullptr);
  if (workerResults_ == nullptr) {
    return nullptr;
  }
  input_ = std::move(pending_.front());
  pending_.pop_front();
  const auto numRows = input_->size();
  VELOX_CHECK_EQ(
      workerResults_->size(),
      numRows,
      "Python worker must return one result per row");
  // Calls outside TRY first: they fail on any bad row, so a call under TRY
  // that shares their results nulls rows only when none is bad.
  for (const bool nullOnError : {false, true}) {
    for (size_t i = 0; i < callColumns_.size(); ++i) {
      if (callColumns_[i].nullOnError == nullOnError) {
        results_[i] = callResults(callColumns_[i], numRows);
      }
    }
  }
  workerResults_ = nullptr;
  auto output = fillOutput(numRows, nullptr);
  input_ = nullptr;
  std::fill(results_.begin(), results_.end(), nullptr);
  return output;
}

VectorPtr PythonCall::callResults(
    const CallColumn& column,
    vector_size_t numRows) {
  const auto index = column.resultIndex;
  auto results = workerResults_->childAt(2 * index);
  const auto& errors = *workerResults_->childAt(2 * index + 1)
                            ->asChecked<SimpleVector<StringView>>();
  if (!column.nullOnError) {
    for (vector_size_t row = 0; row < numRows; ++row) {
      if (!errors.isNullAt(row)) {
        VELOX_USER_FAIL("{}", errors.valueAt(row));
      }
    }
  }
  if (column.containsMap) {
    std::vector<vector_size_t> topRows(numRows);
    std::iota(topRows.begin(), topRows.end(), 0);
    rejectDuplicateMapKeys(
        *results, topRows, *results, column.nullOnError, column.functionName);
  }
  return results;
}

BlockingReason PythonCall::isBlocked(ContinueFuture* future) {
  // Waits only when no more input can go out: the worker has the most
  // batches it may queue, or the input has ended.
  if (pending_.empty() || (pending_.size() < maxPending_ && !noMoreInput_)) {
    return BlockingReason::kNotBlocked;
  }
  tryReceive(future);
  return workerResults_ != nullptr ? BlockingReason::kNotBlocked
                                   : BlockingReason::kWaitForPythonWorker;
}

bool PythonCall::isFinished() {
  return noMoreInput_ && pending_.empty();
}

void PythonCall::close() {
  if (worker_ != nullptr) {
    // A worker that returned every result answers END with its final stats.
    worker_->stop(/*waitForEnd=*/noMoreInput_ && pending_.empty());
    const auto stats = worker_->processStats();
    addRuntimeStat(
        PythonCallStatNames::kWorkerCpuNanos,
        RuntimeCounter(stats.cpuNanos, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        PythonCallStatNames::kWorkerPeakMemoryBytes,
        RuntimeCounter(stats.peakMemoryBytes, RuntimeCounter::Unit::kBytes));
    worker_.reset();
  }
  pending_.clear();
  workerResults_ = nullptr;
  if (workerPool_ != nullptr) {
    workerPool_->release();
    workerPool_.reset();
  }
  Operator::close();
}

} // namespace facebook::velox::exec
