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

#include "velox/core/PlanNode.h"
#include "velox/exec/Linear.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

using ProjectVector =
    std::vector<std::shared_ptr<const core::AbstractProjectNode>>;

struct ITypedExprHasher {
  size_t operator()(const velox::core::ITypedExpr* expr) const {
    return expr->hash();
  }
};

struct ITypedExprComparer {
  bool operator()(
      const velox::core::ITypedExpr* lhs,
      const velox::core::ITypedExpr* rhs) const {
    return *lhs == *rhs;
  }
};

/// Map from leaf expr to OperandIdx. The leaf expr can be a input field
/// reference or stack of struct field getters, an named intermediate or
/// subfield thereof.
using ExprOperandMap = folly::F14FastMap<
    const velox::core::ITypedExpr*,
    OperandIdx,
    ITypedExprHasher,
    ITypedExprComparer>;

/// Describes the input and output of a stage in ProjectSequence.
struct StageData {
  /// Operand idx for each set path in the output of this stage.
  std::vector<Assignment> output;

  /// Expression, 1:1 to 'output'. nullptr if the output is an identity.
  std::vector<core::TypedExprPtr> exprForPath;

  /// OperandIdx for each referenced path in the input of this stage.
  std::vector<Assignment>* input;
  int32_t inputSourceIdx;
  int32_t outputSourceIdx;

  /// Map from getters to
  /// OperandIdx for each distinct path of field getters in an expr of this
  /// stage.
  ExprOperandMap fieldToOperand;

  OperandIdx fieldIdx(const core::TypedExprPtr& field) {
    auto it = fieldToOperand.find(field.get());
    VELOX_CHECK(it != fieldToOperand.end());
    return it->second;
  }

  /// Gathers all distinct OperandIdx for field inputs in the given expression.
  std::vector<OperandIdx> gatherInputs(const core::TypedExprPtr& expr);
};

//// Describes a piece of ExprProgram that produces a single value.
struct ExprInfo {
  /// Fies instruction dx for block.
  int32_t begin;

  /// First instruction idx after block.
  int32_t end;

  /// All inputs that are touched by the block.
  std::vector<OperandIdx> inputs;

  /// The value computed by instructions from begin to end.
  OperandIdx result;
};

// A unit of potentially parallel work. If the same stage has multiple units,
// the inputs, temporary results and results of these units must be
// non-overlapping.
struct WorkUnit {
  // Positions of potential lazies in 'state_' which are to be loaded by this
  // group.
  std::vector<OperandIdx> toLoad;
  std::unique_ptr<core::ExecCtx> execCtx;
  std::unique_ptr<ExprProgram> program;
  std::vector<ExprInfo> programExprs;
  RunState runState;
  int32_t maxNesting{0};
};

class ProjectSequence : public Operator {
 public:
  ProjectSequence(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const ProjectVector& projects);

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool startDrain() override {
    // No need to drain for project/filter operator.
    return false;
  }

  bool isFinished() override;

  void close() override {
    Operator::close();
    work_.clear();
    state_.clear();
    input_.reset();
    results_.clear();
    tempVectors_.clear();
  }

  void initialize() override;

  OperandIdx firstTempIdx() const {
    return firstTempIdx_;
  }

  std::vector<TypePtr>& tempTypes() {
    return tempTypes_;
  }

  const std::vector<TypePtr>& tempTypes() const {
    return tempTypes_;
  }

  ExprOperandMap& constants() {
    return constants_;
  }

  const ExprOperandMap& constants() const {
    return constants_;
  }

  int32_t& stateCounter() {
    return stateCounter_;
  }

  std::vector<VectorPtr>& tempVectors() {
    return tempVectors_;
  }

  core::TypedExprPtr preprocess(const core::TypedExprPtr& tree);

  core::TypedExprPtr tryFoldConstant(const core::TypedExprPtr& expr);

  std::optional<OperandIdx> findReusableInput(
      const core::TypedExprPtr& expr,
      StageData& stage);

  void setConstantValueInfo(const core::TypedExprPtr& constant);

  core::TypedExprPtr setCallValueInfo(const core::TypedExprPtr& call);

  core::TypedExprPtr setCastValueInfo(const core::TypedExprPtr& cast);

  ValueInfoMap& valueMap() {
    return valueMap_;
  }

  void addTempExpr(const core::TypedExprPtr& expr) {
    tempExprs_.push_back(expr);
  }

  OperandIdx makeConstant(const core::TypedExprPtr& expr);

  const RowTypePtr& stageInputType() const {
    return stageInputType_;
  }

  std::string explainExprs() const;

  std::string explainPrograms() const;

 private:
  struct WorkResult {
    WorkResult() : error(nullptr) {}
    WorkResult(std::exception_ptr e) : error(std::move(e)) {}
    std::exception_ptr error;
  };

  std::unique_ptr<WorkResult> runWork(WorkUnit& unit);

  void runStage(int32_t idx);

  void markExprFields(
      const core::TypedExprPtr& expr,
      OperandIdx target,
      StageData& state);

  void makeExprStageData(
      const core::TypedExprPtr& expr,
      std::vector<int32_t>& path,
      StageData& state);

  void makeRowStageData(
      const std::vector<core::TypedExprPtr>& exprs,
      StageData& state);

  void makeWorkUnits(int32_t stageIdx);

  void setLeafRow(
      std::vector<Assignment>& assignments,
      const RowVectorPtr& row);

  void initState(
      const std::vector<Assignment>& assignments,
      const RowVectorPtr& row,
      bool force);

  void setState();

  // Set of consecutive projections.
  ProjectVector projects_;
  RowTypePtr inputType_;
  bool initialized_{false};

  // Maps from paths in inputType_ to indices in state_.
  std::vector<Assignment> inputAssignments_;
  std::vector<StageData> stages_;

  // Expressions to evaluate. All the WorkUnits in the first inner vector are
  // done, then all in the next one. Elements in an inner vector can be
  // parallel. {{a, b}{c, d}} will run a and b possibly in paralel, then after
  // both are done will run c and d, possibly in parallel and then producproduce
  // a result.
  std::vector<std::vector<WorkUnit>> work_;

  // State for all projections. Instructions reference
  // inputs/temporaries/outputs/constants via an index into this.
  std::vector<VectorPtr*> state_;
  // Result vector for each stage.
  std::vector<RowVectorPtr> results_;

  OperandIdx firstTempIdx_{0};
  std::vector<Assignment> temp_;
  std::vector<TypePtr> tempTypes_;
  std::vector<VectorPtr> tempVectors_;
  int32_t stateCounter_{0};
  ValueInfoMap valueMap_;
  ExprOperandMap constants_;

  std::unique_ptr<SimpleExpressionEvaluator> evaluator_;
  std::vector<core::TypedExprPtr> tempExprs_;

  // The type of input for the stage being preprocessed.
  RowTypePtr stageInputType_;

  // Corresponds 1:1 to children of 'stageInputType_'.
  ValueInfo stageInputValueInfo_;

  // ValueInfo for each stage's output. One entry per project in projects_.
  std::vector<ValueInfo> stageValueInfos_;
  bool firstBatch_{true};
};

struct TypeHasher {
  size_t operator()(const velox::TypePtr& type) const {
    // hash on recursive TypeKind. Structs that differ in field names
    // only or decimals with different precisions will collide, no
    // other collisions expected.
    return type->hashKind();
  }
};

struct TypeComparer {
  bool operator()(const velox::TypePtr& lhs, const velox::TypePtr& rhs) const {
    return *lhs == *rhs;
  }
};

/// State during conversion from TypedExpr to ExprProgram
class TranslateCtx {
 public:
  TranslateCtx(StageData& stage, ProjectSequence* projectSequence);

  OperandIdx translateExpr(
      const core::TypedExprPtr&,
      std::optional<OperandIdx> result);

  void setProgram(ExprProgram* program) {
    program_ = program;
  }

  template <typename Instruction, typename... Args>
  void addInstruction(Args&&... args) {
    program_->instructions().push_back(
        std::make_unique<Instruction>(std::forward<Args>(args)...));
  }

  void noReuseOfTemp() {
    for (auto i : distinctTemps_) {
      usedTemps_.insert(i);
    }
  }

  void releaseTemps();

  void allNewTemps();

  std::vector<OperandIdx> gatherNullableInputs(const core::TypedExprPtr& expr);

  int32_t maxNesting() const {
    return maxNesting_ + 1;
  }

  void clearMaxNesting() {
    maxNesting_ = 0;
  }

  OperandIdx makeSwitch(
      const TypePtr& type,
      std::vector<core::TypedExprPtr>& inputs,
      std::optional<OperandIdx> result);

  OperandIdx getTemp(const TypePtr& type);

 private:
  void releaseTemp(OperandIdx idx);
  OperandIdx makeCall(
      const std::string& name,
      const TypePtr& type,
      const std::vector<core::TypedExprPtr>& inputs,
      std::optional<OperandIdx> result);

  void enterNested() {
    nestingLevel_++;
    if (nestingLevel_ > maxNesting_) {
      maxNesting_ = nestingLevel_;
    }
  }

  void leaveNested() {
    nestingLevel_--;
    VELOX_CHECK_GE(nestingLevel_, 0, "Nesting level cannot be negative");
  }

  RowTypePtr inputType_;

  // Operands are checked non-nul for active rows.
  bool inNullPropagating_{false};

  // If inside a conditional, specifies the operand with the flag. The flag of
  // the outermost if is element 1, the flag of the next inner is 1 etc.
  std::vector<OperandIdx> conditions_;

  StageData& stage_;

  /// Map from type to operand index for temporary variables. A temp is a vector
  /// that is in none of   input, named intermediate  or final output.
  std::unordered_map<TypePtr, std::vector<OperandIdx>, TypeHasher, TypeComparer>
      tempVectors_;

  std::unordered_set<OperandIdx> distinctTemps_;

  // Set of possibly fre temps that are used in some parallel activity. parallel
  // work units must have separate temps.
  std::unordered_set<OperandIdx> usedTemps_;
  ProjectSequence* projectSequence_;

  ExprProgram* program_{nullptr};

  // Tracks the maximum nesting level reached during translation
  int32_t maxNesting_{0};

  // Tracks the current nesting level
  int32_t nestingLevel_{0};
};

} // namespace facebook::velox::exec
