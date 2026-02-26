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

#include "velox/exec/ProjectSequence.h"
#include "velox/core/Expressions.h"
#include "velox/exec/Linear.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::exec {

// Creates a chain of FieldAccessTypedExpr for the given path starting from an
// InputTypedExpr. For each index in path, creates a FieldAccessTypedExpr that
// accesses the field at that index from the current row type.
core::TypedExprPtr getterForPath(
    const RowTypePtr& rowType,
    const std::vector<int32_t>& path) {
  // Start with an input reference
  core::TypedExprPtr current = std::make_shared<core::InputTypedExpr>(rowType);

  auto currentRowType = rowType;

  // For each index in path, add a FieldAccessTypedExpr
  for (auto idx : path) {
    auto fieldName = currentRowType->nameOf(idx);
    auto fieldType = currentRowType->childAt(idx);

    // Create FieldAccessTypedExpr with current as input
    current = std::make_shared<core::FieldAccessTypedExpr>(
        fieldType, current, fieldName);

    // If the field type is a ROW, update currentRowType for next iteration
    if (fieldType->kind() == TypeKind::ROW) {
      currentRowType = std::dynamic_pointer_cast<const RowType>(fieldType);
    }
  }

  return current;
}

VectorPtr* addressOfPath(
    const RowVectorPtr& row,
    const std::vector<int32_t>& path) {
  VELOX_CHECK(!path.empty(), "Path must have at least one element");

  RowVector* currentRow = row.get();

  // Navigate through all but the last element of the path
  for (size_t i = 0; i < path.size() - 1; ++i) {
    currentRow = currentRow->childAt(path[i])->as<RowVector>();
  }

  // Return address of the child at the last path index
  return &currentRow->children()[path.back()];
}

std::vector<OperandIdx> StageData::gatherInputs(
    const core::TypedExprPtr& expr) {
  std::unordered_set<OperandIdx> distinctInputs;
  std::vector<int32_t> path;

  // Helper function to recursively collect field inputs
  std::function<void(const core::TypedExprPtr&)> collectFields =
      [&](const core::TypedExprPtr& e) {
        if (!e) {
          return;
        }

        // Check if this expression is a field
        if (isField(e, path)) {
          auto it = fieldToOperand.find(e.get());
          if (it != fieldToOperand.end()) {
            // Extract the actual operand index (remove kMultiple flag if
            // present)
            distinctInputs.insert(OperandIdx(it->second & ~kMultiple));
          }
        }

        // Recursively process all child expressions
        for (const auto& input : e->inputs()) {
          collectFields(input);
        }
      };

  collectFields(expr);

  // Convert set to vector
  return std::vector<OperandIdx>(distinctInputs.begin(), distinctInputs.end());
}

std::optional<OperandIdx> ProjectSequence::findReusableInput(
    const core::TypedExprPtr& expr,
    StageData& stage) {
  if (!expr) {
    return std::nullopt;
  }

  auto kind = expr->kind();

  // Handle field access
  if (kind == core::ExprKind::kFieldAccess ||
      kind == core::ExprKind::kDereference) {
    std::vector<int32_t> path;
    if (isField(expr, path)) {
      auto it = stage.fieldToOperand.find(expr.get());
      if (it != stage.fieldToOperand.end() && isOnlyUse(it->second)) {
        return operandIdx(it->second);
      }
    }
    return std::nullopt;
  }

  // Handle constant
  if (kind == core::ExprKind::kConstant) {
    return std::nullopt;
  }

  // Handle cast
  if (kind == core::ExprKind::kCast) {
    return std::nullopt;
  }

  // Handle call
  if (kind == core::ExprKind::kCall) {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    auto& inputs = call->inputs();

    // Check if this is an "if" or "switch" function first
    // These should be treated specially regardless of metadata
    if (call->name() == "if" || call->name() == "switch") {
      // For if/switch, only consider odd indices and the last arg
      for (int i = 1; i < inputs.size(); i += 2) {
        auto result = findReusableInput(inputs[i], stage);
        if (result.has_value()) {
          return result;
        }
      }
      // Also consider the last arg even if it's at an even index
      if (inputs.size() % 2 == 0 && !inputs.empty()) {
        auto result = findReusableInput(inputs.back(), stage);
        if (result.has_value()) {
          return result;
        }
      }
      return std::nullopt;
    }

    // For other functions, use metadata
    auto metadata = linearMetadata(call->name());

    if (metadata.mayMoveArgToResult) {
      // Check all arguments
      for (const auto& input : inputs) {
        auto result = findReusableInput(input, stage);
        if (result.has_value()) {
          return result;
        }
      }
      return std::nullopt;
    }

    // Check if metadata has maybeMovedArg set
    if (metadata.maybeMovedArg.has_value()) {
      int32_t argIndex = metadata.maybeMovedArg.value();
      if (argIndex >= 0 && argIndex < inputs.size()) {
        return findReusableInput(inputs[argIndex], stage);
      }
    }

    return std::nullopt;
  }

  // All other cases
  return std::nullopt;
}

void ProjectSequence::markExprFields(
    const core::TypedExprPtr& expr,
    OperandIdx target,
    StageData& state) {
  auto kind = expr->kind();
  if (kind == core::ExprKind::kFieldAccess ||
      kind == core::ExprKind::kDereference) {
    std::vector<int32_t> path;
    if (isField(expr, path)) {
      auto it = state.fieldToOperand.find(expr.get());
      OperandIdx inputIdx = kNoOperand;
      if (it == state.fieldToOperand.end()) {
        inputIdx = target == kNoOperand ? stateCounter_++ : target;
        state.fieldToOperand[expr.get()] = inputIdx;
        state.input->emplace_back(path, inputIdx, state.inputSourceIdx);
      } else {
        inputIdx = OperandIdx(it->second & ~kMultiple);
        it->second |= kMultiple;
      }
      return;
    }
  }
  if (target != kNoOperand) {
    state.exprForPath.push_back(expr);
  }
  for (auto i = 0; i < expr->inputs().size(); ++i) {
    markExprFields(expr->inputs()[i], kNoOperand, state);
  }
}

void ProjectSequence::makeExprStageData(
    const core::TypedExprPtr& expr,
    std::vector<int32_t>& path,
    StageData& state) {
  if (expression::utils::isCall(expr, "row_constructor")) {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    for (auto i = 0; i < call->inputs().size(); ++i) {
      path.push_back(i);
      makeExprStageData(call->inputs()[i], path, state);
      path.pop_back();
    }
    return;
  }
  auto destination = stateCounter_++;
  state.output.emplace_back(path, destination, state.outputSourceIdx);
  markExprFields(expr, destination, state);
}

void ProjectSequence::makeRowStageData(
    const std::vector<core::TypedExprPtr>& exprs,
    StageData& state) {
  std::vector<int32_t> path;
  if (&state == &stages_.back()) {
    for (auto i = 0; i < exprs.size(); ++i) {
      path.push_back(i);
      makeExprStageData(exprs[i], path, state);
      path.pop_back();
    }
  } else {
    // Non-last stages start with the paths that were accessed by the next
    // stage.
    state.exprForPath.resize(state.output.size());
    for (auto i = 0; i < state.output.size(); ++i) {
      auto& out = state.output[i];
      auto& project = *projects_[&state - &stages_[0]];
      auto expr = exprForPath(project, out.path);
      VELOX_CHECK_NE(out.operand, kNoOperand);
      markExprFields(expr, out.operand, state);
      state.exprForPath[i] = expr;
    }
  }
}

ProjectSequence::ProjectSequence(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const ProjectVector& projects)
    : Operator(
          driverCtx,
          projects.back()->outputType(),
          operatorId,
          projects.front()->id(),
          "ProjectSequence"),
      projects_(projects),
      inputType_(projects_.front()->sources()[0]->outputType()) {}

TranslateCtx::TranslateCtx(StageData& stage, ProjectSequence* projectSequence)
    : stage_(stage), projectSequence_(projectSequence) {
  // Initialize tempVectors_ map based on tempTypes and constants
  auto& tempTypes = projectSequence_->tempTypes();
  auto& constants = projectSequence_->constants();
  auto firstTempIdx = projectSequence_->firstTempIdx();

  for (size_t i = 0; i < tempTypes.size(); ++i) {
    auto operandIdx = firstTempIdx + i;

    // Check if this operandIdx is NOT used by any constant
    bool isUsedByConstant = false;
    for (const auto& pair : constants) {
      if (pair.second == operandIdx) {
        isUsedByConstant = true;
        break;
      }
    }

    if (!isUsedByConstant) {
      // Add this operandIdx to the vector for the corresponding type
      tempVectors_[tempTypes[i]].push_back(operandIdx);
    }
  }
}

OperandIdx TranslateCtx::getTemp(const TypePtr& type) {
  OperandIdx idx;

  // Look up the type in tempVectors_
  auto it = tempVectors_.find(type);
  if (it != tempVectors_.end() && !it->second.empty()) {
    // Found and vector is not empty, pop the last value
    idx = it->second.back();
    it->second.pop_back();
  } else {
    // No match or vector is empty, create new idx
    idx = projectSequence_->stateCounter()++;
    auto firstTempIdx = projectSequence_->firstTempIdx();
    auto& tempTypes = projectSequence_->tempTypes();

    // Add the type to tempTypes at index (idx - firstTempIdx)
    auto tempIndex = idx - firstTempIdx;
    if (tempIndex >= tempTypes.size()) {
      tempTypes.resize(tempIndex + 1);
    }
    tempTypes[tempIndex] = type;
  }

  // Add idx to distinctTemps_ in both cases
  distinctTemps_.insert(idx);

  return idx;
}

void TranslateCtx::releaseTemp(OperandIdx idx) {
  auto firstTempIdx = projectSequence_->firstTempIdx();
  auto& tempTypes = projectSequence_->tempTypes();

  // Look up the type of the idx from tempTypes at index (idx - firstTempIdx)
  auto tempIndex = idx - firstTempIdx;
  VELOX_CHECK_LT(tempIndex, tempTypes.size(), "Invalid temp index for release");

  auto type = tempTypes[tempIndex];

  // Add the idx to the vector in tempVectors_ that corresponds to the type
  tempVectors_[type].push_back(idx);
}

void TranslateCtx::releaseTemps() {
  for (auto idx : distinctTemps_) {
    releaseTemp(idx);
  }
  distinctTemps_.clear();
}

void TranslateCtx::allNewTemps() {
  tempVectors_.clear();
}

std::vector<OperandIdx> TranslateCtx::gatherNullableInputs(
    const core::TypedExprPtr& expr) {
  std::unordered_set<OperandIdx> distinctInputs;
  std::vector<int32_t> path;

  // Helper function to recursively collect nullable field inputs
  std::function<void(const core::TypedExprPtr&)> collectFields =
      [&](const core::TypedExprPtr& e) {
        if (!e) {
          return;
        }

        // Check if this expression is a field
        if (isField(e, path)) {
          auto it = stage_.fieldToOperand.find(e.get());
          if (it != stage_.fieldToOperand.end()) {
            // Check if the field is nullable using ValueInfo
            auto* info = valueInfo(e.get(), projectSequence_->valueMap());
            if (info && !info->notNull) {
              // Field is nullable, add it to the set
              // Extract the actual operand index (remove kMultiple flag if
              // present)
              distinctInputs.insert(OperandIdx(it->second & ~kMultiple));
            }
          }
        }

        // Recursively process all child expressions
        for (const auto& input : e->inputs()) {
          collectFields(input);
        }
      };

  collectFields(expr);

  // Convert set to vector
  return std::vector<OperandIdx>(distinctInputs.begin(), distinctInputs.end());
}

void ProjectSequence::makeWorkUnits(int stageIdx) {
  const core::AbstractProjectNode* project = projects_[stageIdx].get();
  std::vector<std::vector<core::TypedExprPtr>> groups;
  std::vector<WorkUnit> units;
  if (auto* parallel =
          dynamic_cast<const core::ParallelProjectNode*>(project)) {
    groups = parallel->exprGroups();
  } else {
    groups.push_back(project->projections());
  }
  auto& stage = stages_[stageIdx];
  TranslateCtx ctx(stage, this);
  stageInputType_ =
      stageIdx == 0 ? inputType_ : projects_[stageIdx - 1]->outputType();
  int exprIdx = 0;
  for (auto& group : groups) {
    units.emplace_back();
    auto& unit = units.back();
    unit.execCtx = std::make_unique<core::ExecCtx>(
        operatorCtx_->pool(),
        operatorCtx_->driverCtx()->task->queryCtx().get());
    unit.program = std::make_unique<ExprProgram>();
    ctx.setProgram(unit.program.get());
    for (auto i = 0; i < group.size(); ++i) {
      auto expr = stage.exprForPath[exprIdx];
      if (expr) {
        // Record begin instruction index
        int32_t begin = unit.program->instructions().size();

        // Translate the expression
        ctx.translateExpr(expr, stage.output[exprIdx].operand);

        // Record end instruction index
        int32_t end = unit.program->instructions().size();

        // Get result operand
        OperandIdx result = stage.output[exprIdx].operand;

        // Gather inputs from the expression
        std::vector<OperandIdx> inputs = stage.gatherInputs(expr);

        // Create and append ExprInfo
        ExprInfo exprInfo;
        exprInfo.begin = begin;
        exprInfo.end = end;
        exprInfo.inputs = std::move(inputs);
        exprInfo.result = result;
        unit.programExprs.push_back(exprInfo);

        // Release temps used by this expression
        ctx.releaseTemps();
      }
      ++exprIdx;
    }
    // Set maxNesting from TranslateCtx after translating all expressions
    unit.maxNesting = ctx.maxNesting();
    // Clear temp vectors for the next WorkUnit
    ctx.allNewTemps();
    ctx.noReuseOfTemp();
    ctx.clearMaxNesting();
  }
  work_.push_back(std::move(units));
}

int32_t findPrefixIdx(
    const std::vector<std::vector<int32_t>>& paths,
    const std::vector<int32_t>& path) {
  std::vector<int32_t> prefix;
  prefix.insert(prefix.end(), path.begin(), path.end() - 1);
  auto it = std::find(paths.begin(), paths.end(), prefix);
  VELOX_CHECK(it != paths.end());
  return it - paths.begin();
}

void ProjectSequence::setLeafRow(
    std::vector<Assignment>& assignments,
    const RowVectorPtr& row) {
  for (auto& assignment : assignments) {
    if (state_[assignment.operand]) {
      continue;
    }
    state_[assignment.operand] = addressOfPath(row, assignment.path);
  }
}

void ProjectSequence::initialize() {
  Operator::initialize();

  // Preprocess all expressions in each project
  for (int i = 0; i < projects_.size(); ++i) {
    if (i == 0) {
      stageInputType_ = inputType_;
      stageInputValueInfo_ = makeDefaultValueInfo(inputType_);
    } else {
      stageInputType_ = projects_[i - 1]->outputType();
      // stageInputValueInfo_ was already set at the end of the previous
      // iteration
    }

    // Create nextValueInfo for accumulating this stage's output
    ValueInfo nextValueInfo(true, false);

    // Check if this is a ParallelProjectNode
    if (auto* parallelProject = dynamic_cast<const core::ParallelProjectNode*>(
            projects_[i].get())) {
      // Get mutable access to the parallel project's expression groups
      auto* mutableParallelProject =
          const_cast<core::ParallelProjectNode*>(parallelProject);
      auto& exprGroups =
          const_cast<std::vector<std::vector<core::TypedExprPtr>>&>(
              mutableParallelProject->exprGroups());

      // Preprocess each expression in each group and replace it with the result
      for (auto& group : exprGroups) {
        for (int j = 0; j < group.size(); ++j) {
          group[j] = preprocess(group[j]);
          // Get the ValueInfo for this expression and add as child
          auto* info = valueInfo(group[j].get(), valueMap_);
          if (info) {
            nextValueInfo.children.push_back(*info);
          } else {
            nextValueInfo.children.push_back(ValueInfo(false, false));
          }
        }
      }
    } else {
      // Get mutable access to the project's projections
      auto* mutableProject =
          const_cast<core::AbstractProjectNode*>(projects_[i].get());
      auto& projections = const_cast<std::vector<core::TypedExprPtr>&>(
          mutableProject->projections());

      // Preprocess each expression and replace it with the result
      for (int j = 0; j < projections.size(); ++j) {
        projections[j] = preprocess(projections[j]);
        // Get the ValueInfo for this expression and add as child
        auto* info = valueInfo(projections[j].get(), valueMap_);
        if (info) {
          nextValueInfo.children.push_back(*info);
        } else {
          nextValueInfo.children.push_back(ValueInfo(false, false));
        }
      }
    }

    // Append nextValueInfo to stageValueInfos_
    stageValueInfos_.push_back(nextValueInfo);

    // Set stageInputValueInfo_ for the next stage
    stageInputValueInfo_ = nextValueInfo;
  }

  const auto& inputType = projects_[0]->sources()[0]->outputType();
  stages_.resize(projects_.size());
  for (int32_t level = projects_.size() - 1; level >= 0; --level) {
    if (level == 0) {
      stages_[0].input = &inputAssignments_;
    } else {
      stages_[level].input = &stages_[level - 1].output;
    }
    stages_[level].inputSourceIdx = level;
    stages_[level].outputSourceIdx = level + 1;
    makeRowStageData(projects_[level]->projections(), stages_[level]);
  }
  firstTempIdx_ = stateCounter_;
  for (auto i = 0; i < projects_.size(); ++i) {
    makeWorkUnits(i);
  }
  for (auto& assignment : inputAssignments_) {
    // set to non-0 to mark this will be set on first input.
    state_[assignment.operand] = reinterpret_cast<VectorPtr*>(1);
  }
  for (auto i = 0; i < stages_.size(); ++i) {
    setLeafRow(stages_[i].output, results_[i]);
  }
  tempVectors_.resize(tempTypes_.size());
  for (auto i = 0; i < tempVectors_.size(); ++i) {
    state_[firstTempIdx_ + i] = &tempVectors_[i];
  }
}

std::unique_ptr<ProjectSequence::WorkResult> ProjectSequence::runWork(
    WorkUnit& unit) {
  try {
    // Ensure selectionStack has at least maxNesting SelectivityVectors
    if (unit.runState.selectionStack.size() < unit.maxNesting) {
      unit.runState.selectionStack.resize(unit.maxNesting);
      for (auto& selection : unit.runState.selectionStack) {
        if (!selection) {
          selection = std::make_unique<SelectivityVector>(input_->size());
        }
      }
    }

    // Resize all SelectivityVectors to current input size
    for (auto& selection : unit.runState.selectionStack) {
      selection->resize(input_->size());
      selection->setAll();
    }

    // Reset selection to the first element and mark all rows as active
    unit.runState.resetSelection();
    unit.runState.state = state_.data();

    // Process each ExprInfo
    EvalCtx evalCtx(unit.execCtx.get());
    for (const auto& exprInfo : unit.programExprs) {
      unit.program->eval(&evalCtx, exprInfo.begin, exprInfo.end, unit.runState);
    }

    // Return empty WorkResult (no error)
    return std::make_unique<WorkResult>();
  } catch (...) {
    // Catch any exception and return it in WorkResult
    return std::make_unique<WorkResult>(std::current_exception());
  }
}

void ProjectSequence::addInput(RowVectorPtr input) {
  input_ = std::move(input);
}

bool ProjectSequence::isFinished() {
  return noMoreInput_ && !input_;
}

RowVectorPtr ProjectSequence::getOutput() {
  if (!input_) {
    return nullptr;
  }
  SCOPE_EXIT {
    input_.reset();
  };

  vector_size_t size = input_->size();
  LocalSelectivityVector localRows(*operatorCtx_->execCtx(), size);
  auto* rows = localRows.get();
  VELOX_DCHECK_NOT_NULL(rows);
  rows->setAll();
  setState();

  // Process each stage in work_
  for (auto& stage : work_) {
    if (stage.size() == 1) {
      // Single WorkUnit - run on this thread
      auto& unit = stage[0];
      auto result = runWork(unit);
      if (result->error) {
        std::rethrow_exception(result->error);
      }
    } else if (stage.size() > 1) {
      // Multiple WorkUnits - run in parallel
      std::vector<std::shared_ptr<AsyncSource<WorkResult>>> pending;

      for (auto i = 0; i < stage.size(); ++i) {
        auto& unit = stage[i];
        pending.push_back(
            std::make_shared<AsyncSource<WorkResult>>(
                [this, &unit]() { return runWork(unit); }));
        auto item = pending.back();
        operatorCtx_->task()->queryCtx()->executor()->add(
            [item]() { item->prepare(); });
      }

      // Wait for all parallel work to complete and handle errors
      std::exception_ptr error;
      for (auto i = 0; i < pending.size(); ++i) {
        auto result = pending[i]->move();
        stats_.wlock()->getOutputTiming.add(pending[i]->prepareTiming());
        if (!error && result->error) {
          error = result->error;
        }
      }

      if (error) {
        std::rethrow_exception(error);
      }
    }
  }

  return results_.back();
}

void listRows(
    const RowType* row,
    std::vector<int32_t>& path,
    std::vector<std::vector<int32_t>>& result) {
  for (auto i = 0; i < row->size(); ++i) {
    auto child = row->childAt(i);
    if (child->kind() == TypeKind::ROW) {
      path.push_back(i);
      result.push_back(path);
      listRows(&child->asRow(), path, result);
      path.pop_back();
    }
  }
}

std::vector<VectorPtr>* getRowAt(
    RowVector* row,
    const std::vector<int32_t> path) {
  for (auto idx : path) {
    row = row->childAt(idx)->as<RowVector>();
    VELOX_CHECK_EQ(row->encoding(), VectorEncoding::Simple::ROW);
  }
  return &row->children();
}

void ProjectSequence::initState(
    const std::vector<Assignment>& assignments,
    const RowVectorPtr& row,
    bool force) {
  for (auto& assignment : assignments) {
    if (!force && !state_[assignment.operand]) {
      continue;
    }
    state_[operandIdx(assignment.operand)] =
        addressOfPath(row, assignment.path);
  }
}

void ProjectSequence::setState() {
  state_.resize(stateCounter_);
  initState(inputAssignments_, input_, true);
  if (firstBatch_) {
    firstBatch_ = false;
    for (auto& project : projects_) {
      results_.push_back(
          BaseVector::create<RowVector>(
              project->outputType(), input_->size(), operatorCtx_->pool()));
    }
    for (auto i = 0; i < stages_.size(); ++i) {
      auto& stage = stages_[i];
      initState(stage.output, results_[i], false);
    }
  }
}

} // namespace facebook::velox::exec
