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

#include "velox/exec/Linear.h"
#include "velox/core/Expressions.h"
#include "velox/exec/ProjectSequence.h"
#include "velox/exec/Task.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::exec {

namespace {

const auto& opCodeNames() {
  static const folly::F14FastMap<Instruction::OpCode, std::string_view> kNames =
      {
          {Instruction::OpCode::kNulls, "kNulls"},
          {Instruction::OpCode::kNullsEnd, "kNullsEnd"},
          {Instruction::OpCode::kIf, "kIf"},
          {Instruction::OpCode::kCoalesce, "kCoalesce"},
          {Instruction::OpCode::kCall, "kCall"},
          {Instruction::OpCode::kField, "kField"},
          {Instruction::OpCode::kAssign, "kAssign"},
      };
  return kNames;
}

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(Instruction, OpCode, opCodeNames)

Call::Call(
    OperandIdx result,
    std::vector<OperandIdx> args,
    TypePtr type,
    int32_t returnedArg,
    std::shared_ptr<VectorFunction> vectorFunction,
    VectorFunctionMetadata vectorFunctionMetadata)
    : Instruction(OpCode::kCall, result),
      args_(std::move(args)),
      vectorFunction_(std::move(vectorFunction)),
      vectorFunctionMetadata_(std::move(vectorFunctionMetadata)),
      type_(std::move(type)),
      returnedArg_(returnedArg) {
  // For each arg, check if there's a previous arg with the same operandIdx()
  // If so, OR kCopyPtr to the non-first occurrence
  for (size_t i = 1; i < args_.size(); ++i) {
    auto currentIdx = operandIdx(args_[i]);
    for (size_t j = 0; j < i; ++j) {
      if (operandIdx(args_[j]) == currentIdx) {
        // Found a duplicate, set kCopyPtr bit on current arg
        args_[i] |= kCopyPtr;
        break;
      }
    }
  }
}

std::string Instruction::toString() const {
  return fmt::format("{} result={}\n", toName(opCode_), result_);
}

std::string Field::toString() const {
  return fmt::format(
      "{} result={} input={} childIdx={}\n",
      toName(opCode_),
      result_,
      input_,
      childIdx_);
}

std::string Assign::toString() const {
  return fmt::format(
      "{} result={} source={}\n", toName(opCode_), result_, source_);
}

std::string If::toString() const {
  return fmt::format(
      "{} condition={} elseIdx={} endIdx={}\n",
      toName(opCode_),
      condition_,
      elseIdx_,
      endIdx_);
}

std::string Nulls::toString() const {
  std::string operandsStr;
  for (size_t i = 0; i < operands_.size(); ++i) {
    if (i > 0) {
      operandsStr += ",";
    }
    operandsStr += std::to_string(operands_[i]);
  }
  return fmt::format(
      "{} result={} operands=[{}] nullFlagIdx={}\n",
      toName(opCode_),
      result_,
      operandsStr,
      nullFlagIdx_);
}

std::string NullsEnd::toString() const {
  return fmt::format("{} result={}\n", toName(opCode_), result_);
}

std::string Coalesce::toString() const {
  return fmt::format(
      "{} result={} input={} default={}\n",
      toName(opCode_),
      result_,
      input_,
      default_);
}

std::string Call::toString() const {
  std::string argsStr;
  for (size_t i = 0; i < args_.size(); ++i) {
    if (i > 0) {
      argsStr += ",";
    }
    argsStr += std::to_string(args_[i]);
  }
  return fmt::format(
      "{} result={} args=[{}] returnedArg={}\n",
      toName(opCode_),
      result_,
      argsStr,
      returnedArg_);
}

namespace {

// Helper function to resolve vector function and metadata for linear execution
// Returns a pair of (VectorFunction, VectorFunctionMetadata)
std::pair<std::shared_ptr<VectorFunction>, VectorFunctionMetadata>
resolveVectorFunctionForLinear(
    const std::string& name,
    const std::vector<TypePtr>& inputTypes,
    const core::QueryConfig& config) {
  // Create empty constant inputs vector (all nullptr) aligned with inputTypes
  std::vector<VectorPtr> constantInputs(inputTypes.size(), nullptr);

  // First try to get vector function with metadata from vector function
  // registry
  if (auto functionWithMetadata = getVectorFunctionWithMetadata(
          name, inputTypes, constantInputs, config)) {
    return *functionWithMetadata;
  }

  // Then try simple function registry
  if (auto simpleFunctionEntry =
          simpleFunctions().resolveFunction(name, inputTypes)) {
    auto func = simpleFunctionEntry->createFunction()->createVectorFunction(
        inputTypes, constantInputs, config);
    return {std::move(func), simpleFunctionEntry->metadata()};
  }

  // If neither registry has the function, throw an error
  VELOX_USER_FAIL(
      "Scalar function name not registered: {}, called with arguments: ({}).",
      name,
      folly::join(", ", inputTypes));
}

// Returns a reference to the static map of linear metadata.
std::unordered_map<std::string, FunctionLinearMetadata>&
getLinearMetadataMap() {
  static std::unordered_map<std::string, FunctionLinearMetadata> metadataMap;
  return metadataMap;
}

// Returns a reference to the static map of special forms.
std::unordered_map<std::string, TranslateSpecialForm>& getSpecialFormMap() {
  static std::unordered_map<std::string, TranslateSpecialForm> specialFormMap;
  return specialFormMap;
}

} // namespace

core::TypedExprPtr copyWithChildren(
    const core::TypedExprPtr& expr,
    const std::vector<core::TypedExprPtr>& newChildren) {
  switch (expr->kind()) {
    case core::ExprKind::kCall: {
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      return std::make_shared<core::CallTypedExpr>(
          call->type(), newChildren, call->name());
    }
    case core::ExprKind::kCast: {
      auto cast = expr->asUnchecked<core::CastTypedExpr>();
      return std::make_shared<core::CastTypedExpr>(
          cast->type(), newChildren, cast->isTryCast());
    }
    case core::ExprKind::kDereference: {
      auto deref = expr->asUnchecked<core::DereferenceTypedExpr>();
      VELOX_CHECK_EQ(
          newChildren.size(),
          1,
          "DereferenceTypedExpr requires exactly one child");
      return std::make_shared<core::DereferenceTypedExpr>(
          deref->type(), newChildren[0], deref->index());
    }
    case core::ExprKind::kFieldAccess: {
      auto field = expr->asUnchecked<core::FieldAccessTypedExpr>();
      if (field->isInputColumn()) {
        // Input column field access has no children
        VELOX_CHECK(
            newChildren.empty(),
            "Input column FieldAccessTypedExpr should have no children");
        return std::make_shared<core::FieldAccessTypedExpr>(
            field->type(), field->name());
      } else {
        // Struct field access has one child
        VELOX_CHECK_EQ(
            newChildren.size(),
            1,
            "Struct FieldAccessTypedExpr requires exactly one child");
        return std::make_shared<core::FieldAccessTypedExpr>(
            field->type(), newChildren[0], field->name());
      }
    }
    default:
      VELOX_UNSUPPORTED(
          "copyWithChildren not implemented for expression kind: {}",
          static_cast<int32_t>(expr->kind()));
  }
}

const ValueInfo* valueInfo(
    const core::ITypedExpr* expr,
    const ValueInfoMap& map) {
  auto it = map.find(expr);
  return it == map.end() ? nullptr : &it->second;
}

ValueInfo vectorValueInfo(const BaseVector& vector) {
  auto encoding = vector.encoding();
  switch (encoding) {
    case VectorEncoding::Simple::CONSTANT: {
      if (vector.isNullAt(0)) {
        return ValueInfo(false, false);
      }
      auto* wrapped = vector.wrappedVector();
      if (wrapped == &vector) {
        return ValueInfo(true, true);
      }

      return vectorValueInfo(*wrapped);
    }
    case VectorEncoding::Simple::FLAT:
      return ValueInfo(!vector.mayHaveNulls(), !vector.mayHaveNulls());
    case VectorEncoding::Simple::DICTIONARY:
      return vectorValueInfo(*vector.wrappedVector());
    case VectorEncoding::Simple::ROW: {
      std::vector<ValueInfo> childInfo;
      bool allNotNull = true;
      for (auto& child : vector.as<RowVector>()->children()) {
        childInfo.push_back(vectorValueInfo(*child));
        allNotNull &= childInfo.back().recursiveNotNull;
      }
      return ValueInfo(true, allNotNull, std::move(childInfo));
    }
    case VectorEncoding::Simple::ARRAY: {
      std::vector<ValueInfo> childInfo = {
          vectorValueInfo(*vector.as<ArrayVector>()->elements())};
      return ValueInfo(
          true, childInfo[0].recursiveNotNull, std::move(childInfo));
    }
    case VectorEncoding::Simple::MAP: {
      std::vector<ValueInfo> childInfo = {
          vectorValueInfo(*vector.as<MapVector>()->mapKeys()),
          vectorValueInfo(*vector.as<MapVector>()->mapValues())};
      bool recursiveNotNull =
          childInfo[0].recursiveNotNull && childInfo[1].recursiveNotNull;
      return ValueInfo(true, recursiveNotNull, std::move(childInfo));
    }
    default:
      VELOX_FAIL("Unsupported encoding {}", encoding);
  }
}

ValueInfo makeEmptyValueInfo(const TypePtr& type) {
  std::vector<ValueInfo> children;

  if (type->kind() == TypeKind::ROW) {
    auto& rowType = type->asRow();
    for (int i = 0; i < rowType.size(); ++i) {
      children.push_back(makeEmptyValueInfo(rowType.childAt(i)));
    }
  } else if (type->kind() == TypeKind::ARRAY) {
    children.push_back(makeEmptyValueInfo(type->asArray().elementType()));
  } else if (type->kind() == TypeKind::MAP) {
    children.push_back(makeEmptyValueInfo(type->asMap().keyType()));
    children.push_back(makeEmptyValueInfo(type->asMap().valueType()));
  }

  return ValueInfo(true, true, std::move(children));
}

ValueInfo makeDefaultValueInfo(const TypePtr& type) {
  std::vector<ValueInfo> children;

  if (type->kind() == TypeKind::ROW) {
    auto& rowType = type->asRow();
    for (int i = 0; i < rowType.size(); ++i) {
      children.push_back(makeDefaultValueInfo(rowType.childAt(i)));
    }
  } else if (type->kind() == TypeKind::ARRAY) {
    children.push_back(makeDefaultValueInfo(type->asArray().elementType()));
  } else if (type->kind() == TypeKind::MAP) {
    children.push_back(makeDefaultValueInfo(type->asMap().keyType()));
    children.push_back(makeDefaultValueInfo(type->asMap().valueType()));
  }

  return ValueInfo(false, false, std::move(children));
}

void mergeValueInfo(const ValueInfo other, ValueInfo& result) {
  result.notNull = result.notNull && other.notNull;
  result.recursiveNotNull = result.recursiveNotNull && other.recursiveNotNull;

  VELOX_CHECK_EQ(result.children.size(), other.children.size());
  for (size_t i = 0; i < result.children.size(); ++i) {
    mergeValueInfo(other.children[i], result.children[i]);
  }
}

void ProjectSequence::setConstantValueInfo(const core::TypedExprPtr& constant) {
  auto constantExpr = constant->asUnchecked<core::ConstantTypedExpr>();

  VectorPtr vector;
  if (constantExpr->hasValueVector()) {
    vector = constantExpr->valueVector();
  } else {
    vector = BaseVector::createConstant(
        constantExpr->type(), constantExpr->value(), 1, operatorCtx()->pool());
  }

  auto info = vectorValueInfo(*vector);
  valueMap_[constant.get()] = std::move(info);
}

OperandIdx ProjectSequence::makeConstant(const core::TypedExprPtr& expr) {
  // Check if expr is already in constants_
  auto it = constants_.find(expr.get());
  if (it != constants_.end()) {
    return it->second;
  }

  // Add expr to tempExprs_
  addTempExpr(expr);

  // Get new OperandIdx
  OperandIdx idx = stateCounter_++;

  // Add to constants_
  constants_[expr.get()] = idx;

  // Make sure state_ has at least idx + 1 elements
  if (state_.size() <= idx) {
    state_.resize(idx + 1);
  }

  // Create the ConstantVector
  auto constantExpr = expr->asUnchecked<core::ConstantTypedExpr>();
  VectorPtr vector;
  if (constantExpr->hasValueVector()) {
    vector = constantExpr->valueVector();
  } else {
    vector = BaseVector::createConstant(
        constantExpr->type(), constantExpr->value(), 1, operatorCtx()->pool());
  }

  // Add to tempVectors_
  if (tempVectors_.size() <= idx) {
    tempVectors_.resize(idx + 1);
  }
  tempVectors_[idx] = vector;

  // Set state_[idx] to point to the vector
  state_[idx] = &tempVectors_[idx];

  return idx;
}

std::string ProjectSequence::explainExprs() const {
  std::string result;

  for (size_t i = 0; i < projects_.size(); ++i) {
    const auto& project = projects_[i];

    // Check if this is a ParallelProjectNode
    if (auto* parallelProject =
            dynamic_cast<const core::ParallelProjectNode*>(project.get())) {
      result += fmt::format("Project {}: parallel project\n", i);

      const auto& exprGroups = parallelProject->exprGroups();
      const auto& names = parallelProject->exprNames();

      size_t nameIdx = 0;
      for (size_t groupIdx = 0; groupIdx < exprGroups.size(); ++groupIdx) {
        result += fmt::format("  Group {}:\n", groupIdx);
        const auto& group = exprGroups[groupIdx];

        for (const auto& expr : group) {
          if (nameIdx < names.size()) {
            result +=
                fmt::format("    {}: {}\n", names[nameIdx], expr->toString());
            ++nameIdx;
          }
        }
      }
    } else {
      // Regular project
      result += fmt::format("Project {}: project\n", i);

      const auto& names = project->names();
      const auto& projections = project->projections();

      for (size_t j = 0; j < projections.size(); ++j) {
        result +=
            fmt::format("  {}: {}\n", names[j], projections[j]->toString());
      }
    }
  }

  return result;
}

std::string ProjectSequence::explainPrograms() const {
  std::string result;

  for (size_t i = 0; i < work_.size(); ++i) {
    result += fmt::format("Stage {}:\n", i);

    const auto& units = work_[i];
    for (size_t unitIdx = 0; unitIdx < units.size(); ++unitIdx) {
      const auto& unit = units[unitIdx];
      result += fmt::format("  WorkUnit {}:\n", unitIdx);

      for (size_t exprIdx = 0; exprIdx < unit.programExprs.size(); ++exprIdx) {
        const auto& exprInfo = unit.programExprs[exprIdx];

        // Format inputs as comma-separated list
        std::string inputsStr;
        for (size_t j = 0; j < exprInfo.inputs.size(); ++j) {
          if (j > 0) {
            inputsStr += ", ";
          }
          inputsStr += std::to_string(exprInfo.inputs[j]);
        }

        result += fmt::format(
            "    ExprInfo {}: result={} inputs=[{}]\n",
            exprIdx,
            exprInfo.result,
            inputsStr);

        // Print instructions from begin to end (exclusive)
        for (int32_t instrIdx = exprInfo.begin; instrIdx < exprInfo.end;
             ++instrIdx) {
          const auto& instr = unit.program->instructions()[instrIdx];
          result += fmt::format("      {}: {}", instrIdx, instr->toString());
        }
      }
    }
  }

  return result;
}

core::TypedExprPtr ProjectSequence::setCallValueInfo(
    const core::TypedExprPtr& call) {
  auto callExpr = call->asUnchecked<core::CallTypedExpr>();
  auto md = linearMetadata(callExpr->name());
  ValueCtx ctx{valueMap_};
  if (md.rewrite) {
    auto rewritten = md.rewrite(call, ctx);
    if (rewritten != call) {
      return preprocess(rewritten);
    }
  }
  ValueInfo info(false, false);

  // If the function has custom makeValueInfo, use it
  if (md.makeValueInfo) {
    info = md.makeValueInfo(call, ctx);
  } else {
    // Check if function has default null behavior
    auto functionMetadata = getVectorFunctionMetadata(callExpr->name());

    if (functionMetadata && functionMetadata->defaultNullBehavior) {
      // Result is not null if all arguments are not null
      bool allNotNull = true;
      for (const auto& input : callExpr->inputs()) {
        auto* inputInfo = valueInfo(input.get(), ctx.valueInfo);
        if (!inputInfo || !inputInfo->notNull) {
          allNotNull = false;
          break;
        }
      }
      info = ValueInfo(allNotNull, allNotNull);

      // Check if all arguments are constants, fields, or calls with
      // allDefaultNullBehavior
      bool allDefaultNullBehavior = true;
      std::vector<int32_t> path;
      for (const auto& input : callExpr->inputs()) {
        bool isValid = false;
        if (input->kind() == core::ExprKind::kConstant) {
          isValid = true;
        } else if (
            input->kind() == core::ExprKind::kFieldAccess &&
            isField(input, path)) {
          isValid = true;
        } else if (input->kind() == core::ExprKind::kCall) {
          auto* inputInfo = valueInfo(input.get(), ctx.valueInfo);
          if (inputInfo && inputInfo->allDefaultNullBehavior) {
            isValid = true;
          }
        }
        if (!isValid) {
          allDefaultNullBehavior = false;
          break;
        }
      }
      info.allDefaultNullBehavior = allDefaultNullBehavior;
    } else {
      // Otherwise, result is nullable
      info = ValueInfo(false, false);
    }
  }

  valueMap_[call.get()] = std::move(info);
  return call;
}

bool isRowRenameCast(const core::CastTypedExpr& cast) {
  if (!expression::utils::isCall(cast.inputs()[0], "row_constructor")) {
    return false;
  }
  return cast.type()->kind() == TypeKind::ROW &&
      cast.type()->equivalent(*cast.inputs()[0]->type());
}

core::TypedExprPtr ProjectSequence::setCastValueInfo(
    const core::TypedExprPtr& cast) {
  auto castExpr = cast->asUnchecked<core::CastTypedExpr>();
  if (isRowRenameCast(*castExpr)) {
    return preprocess(
        std::make_shared<core::CallTypedExpr>(
            castExpr->type(),
            castExpr->inputs()[0]->inputs(),
            "row_constructor"));
  }

  ValueInfo info(false, false);

  if (!castExpr->isTryCast() && !castExpr->inputs().empty()) {
    // If not tryCast, the ValueInfo is nullable if the argument is nullable
    auto* argInfo = valueInfo(castExpr->inputs()[0].get(), valueMap_);
    if (argInfo && argInfo->notNull) {
      info = ValueInfo(true, true);
    }
  }
  // If tryCast is true, result is nullable (default: false, false)

  // Cast has default null behavior, check if argument qualifies for
  // allDefaultNullBehavior
  if (!castExpr->inputs().empty()) {
    auto& input = castExpr->inputs()[0];
    bool allDefaultNullBehavior = false;
    std::vector<int32_t> path;

    if (input->kind() == core::ExprKind::kConstant) {
      allDefaultNullBehavior = true;
    } else if (
        input->kind() == core::ExprKind::kFieldAccess && isField(input, path)) {
      allDefaultNullBehavior = true;
    } else if (input->kind() == core::ExprKind::kCall) {
      auto* inputInfo = valueInfo(input.get(), valueMap_);
      if (inputInfo && inputInfo->allDefaultNullBehavior) {
        allDefaultNullBehavior = true;
      }
    }
    info.allDefaultNullBehavior = allDefaultNullBehavior;
  }

  valueMap_[cast.get()] = std::move(info);
  return cast;
}

core::TypedExprPtr ProjectSequence::tryFoldConstant(
    const core::TypedExprPtr& expr) {
  // Create evaluator if not exists
  if (!evaluator_) {
    evaluator_ = std::make_unique<SimpleExpressionEvaluator>(
        operatorCtx()->driverCtx()->task->queryCtx().get(),
        operatorCtx()->pool());
  }

  try {
    // Try to compile and check if it resulted in a constant
    auto exprSet = evaluator_->compile(expr);
    auto& compiledExprs = exprSet->exprs();

    if (!compiledExprs.empty() && compiledExprs[0]->isConstant()) {
      // The expression was folded to a constant
      auto constantExpr =
          dynamic_cast<const ConstantExpr*>(compiledExprs[0].get());
      if (constantExpr) {
        auto constant =
            std::make_shared<core::ConstantTypedExpr>(constantExpr->value());
        setConstantValueInfo(constant);
        return constant;
      }
    }
  } catch (...) {
    // If constant folding fails, return original
  }
  return expr;
}

core::TypedExprPtr ProjectSequence::preprocess(const core::TypedExprPtr& tree) {
  if (!tree) {
    return tree;
  }

  if (tree->kind() == core::ExprKind::kFieldAccess) {
    auto fieldAccess = tree->asUnchecked<core::FieldAccessTypedExpr>();

    // Check if this is an input field or has input reference as input
    if (fieldAccess->inputs().empty() ||
        fieldAccess->inputs()[0]->kind() == core::ExprKind::kInput) {
      // Find the index of the field in stageInputType_
      auto fieldIdx = stageInputType_->getChildIdx(fieldAccess->name());
      // Set the value info to corresponding element of stageInputValueInfo_
      valueMap_[tree.get()] = stageInputValueInfo_.children[fieldIdx];
      return tree;
    }

    // Otherwise, preprocess the input and extract child value info
    auto input = preprocess(fieldAccess->inputs()[0]);
    auto& inputType = input->type()->asRow();
    auto fieldIdx = inputType.getChildIdx(fieldAccess->name());

    // Get the value info of the input
    auto* inputInfo = valueInfo(input.get(), valueMap_);
    if (inputInfo && fieldIdx < inputInfo->children.size()) {
      valueMap_[tree.get()] = inputInfo->children[fieldIdx];
    } else {
      valueMap_[tree.get()] = ValueInfo(false, false);
    }

    return tree;
  }

  // First, recursively preprocess all children
  std::vector<core::TypedExprPtr> newInputs;
  bool anyChanged = false;

  for (const auto& input : tree->inputs()) {
    auto newInput = preprocess(input);
    if (newInput != input) {
      anyChanged = true;
    }
    newInputs.push_back(newInput);
  }

  // Check if this is a call with all constant arguments
  if (tree->kind() == core::ExprKind::kCall) {
    auto call = tree->asUnchecked<core::CallTypedExpr>();
    bool allConstant = true;

    for (const auto& input : newInputs) {
      if (input->kind() != core::ExprKind::kConstant) {
        allConstant = false;
        break;
      }
    }

    if (allConstant && !newInputs.empty()) {
      // Create the expression with new inputs for constant folding
      auto exprToFold = anyChanged ? std::make_shared<core::CallTypedExpr>(
                                         call->type(), newInputs, call->name())
                                   : tree;
      auto result = tryFoldConstant(exprToFold);
      if (result->kind() == core::ExprKind::kCall) {
        return setCallValueInfo(result);
      } else if (result->kind() == core::ExprKind::kCast) {
        return setCastValueInfo(result);
      }
      return result;
    }
    auto md = linearMetadata(call->name());
    if (md.rewrite) {
      ValueCtx ctx{valueMap_};
      auto rewritten = md.rewrite(tree, ctx);
      if (rewritten != tree) {
        return preprocess(rewritten);
      }
    }
  }

  // Check if this is a cast with constant argument
  if (tree->kind() == core::ExprKind::kCast) {
    if (!newInputs.empty() &&
        newInputs[0]->kind() == core::ExprKind::kConstant) {
      auto cast = tree->asUnchecked<core::CastTypedExpr>();
      auto exprToFold = anyChanged
          ? std::make_shared<core::CastTypedExpr>(
                cast->type(), newInputs, cast->isTryCast())
          : tree;
      auto result = tryFoldConstant(exprToFold);
      if (result->kind() == core::ExprKind::kCall) {
        return setCallValueInfo(result);
      } else if (result->kind() == core::ExprKind::kCast) {
        return setCastValueInfo(result);
      }
      return result;
    }
  }

  // If any inputs changed, create a new expression with the new inputs
  if (anyChanged) {
    auto result = copyWithChildren(tree, newInputs);
    if (result->kind() == core::ExprKind::kCall) {
      setCallValueInfo(result);
    } else if (result->kind() == core::ExprKind::kCast) {
      setCastValueInfo(result);
    }
    return result;
  }

  // No changes, return original
  auto result = tree;
  if (result->kind() == core::ExprKind::kCall) {
    setCallValueInfo(result);
  } else if (result->kind() == core::ExprKind::kCast) {
    setCastValueInfo(result);
  }
  return result;
}

OperandIdx TranslateCtx::makeCall(
    const std::string& name,
    const TypePtr& type,
    const std::vector<core::TypedExprPtr>& inputs,
    std::optional<OperandIdx> result) {
  auto& valueMap = projectSequence_->valueMap();
  auto metadata = linearMetadata(name);
  std::vector<OperandIdx> args;
  OperandIdx resultIdx;

  if (metadata.mayMoveArgToResult) {
    bool reusedInput = false;
    for (auto i = 0; i < inputs.size(); ++i) {
      auto input = inputs[i];
      auto reusable = projectSequence_->findReusableInput(input, stage_);
      if (reusable.has_value()) {
        if (result.has_value() && !reusedInput) {
          args.push_back(translateExpr(input, result));
          reusedInput = true;
          continue;
        }
      }
      args.push_back(translateExpr(input, reusable));
    }
    resultIdx = reusedInput ? result.value() : getTemp(type);
  } else if (metadata.maybeMovedArg.has_value()) {
    auto idx = metadata.maybeMovedArg.value();
    auto moveArg = inputs[idx];
    OperandIdx moveOperand = kNoOperand;
    auto reusable = projectSequence_->findReusableInput(moveArg, stage_);
    if (reusable.has_value() && result.has_value()) {
      moveOperand = translateExpr(moveArg, result);
    } else {
      moveOperand = translateExpr(moveArg, reusable);
    }
    for (auto& input : inputs) {
      if (input == moveArg) {
        args.push_back(moveOperand);
      } else {
        args.push_back(translateExpr(input, std::nullopt));
      }
    }
    resultIdx = moveOperand;
  } else {
    // The result is always not same as any of the args.
    if (!result.has_value()) {
      // Not generating for a specific target.
      resultIdx = getTemp(type);
    } else {
      resultIdx = result.value();
    }
    for (auto input : inputs) {
      auto reusable = projectSequence_->findReusableInput(input, stage_);
      args.push_back(translateExpr(input, reusable));
    }
  }

  // Get input types from the TypedExpr inputs
  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(inputs.size());
  for (const auto& input : inputs) {
    inputTypes.push_back(input->type());
  }

  // Resolve the vector function and metadata
  auto& queryConfig = projectSequence_->operatorCtx()
                          ->driverCtx()
                          ->task->queryCtx()
                          ->queryConfig();
  auto [vectorFunction, vectorFunctionMetadata] =
      resolveVectorFunctionForLinear(name, inputTypes, queryConfig);

  // Determine if the result is the same as one of the arguments
  int32_t returnedArg = -1;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == resultIdx) {
      returnedArg = static_cast<int32_t>(i);
      break;
    }
  }

  // Add the Call instruction
  addInstruction<Call>(
      resultIdx,
      std::move(args),
      type,
      returnedArg,
      std::move(vectorFunction),
      std::move(vectorFunctionMetadata));

  return resultIdx;
}

OperandIdx TranslateCtx::makeSwitch(
    const TypePtr& type,
    std::vector<core::TypedExprPtr>& inputs,
    std::optional<OperandIdx> result) {
  int32_t resultIdx;
  if (!result.has_value()) {
    resultIdx = getTemp(type);
  } else {
    resultIdx = result.value();
  }
  std::vector<If*> ifs;
  for (auto i = 0; i < inputs.size(); i += 2) {
    auto cond = translateExpr(inputs[i], std::nullopt);
    enterNested();
    addInstruction<If>(cond, 0, 0);
    ifs.push_back(program_->instructions().back()->as<If>());
    conditions_.push_back(cond);
    translateExpr(inputs[i + 1], resultIdx);
    ifs.back()->setElse(program_->instructions().size());
    if (i + 2 >= inputs.size()) {
      if (i + 2 == inputs.size()) {
        // No else.
        auto nullValue = Variant(type->kind());
        auto null = projectSequence_->makeConstant(
            std::make_shared<core::ConstantTypedExpr>(type, nullValue));
        addInstruction<Assign>(resultIdx, null);
      } else {
        // There is an else.
        translateExpr(inputs[i + 2], resultIdx);
      }
      // Set each if to end after the else.
      for (auto* ifInst : ifs) {
        leaveNested();
        ifInst->setEnd(program_->instructions().size());
      }
      conditions_.pop_back();
      break;
    }
    conditions_.pop_back();
  }
  return resultIdx;
}

namespace {
Assignment* findAssignment(
    std::vector<Assignment>* assignments,
    const std::vector<int32_t>& path) {
  if (!assignments) {
    return nullptr;
  }
  for (auto& assignment : *assignments) {
    if (assignment.path == path) {
      return &assignment;
    }
  }
  return nullptr;
}
} // namespace

OperandIdx TranslateCtx::translateExpr(
    const core::TypedExprPtr& expr,
    std::optional<OperandIdx> result) {
  // Check if we should add null propagation
  if (!inNullPropagating_) {
    auto* info = valueInfo(expr.get(), projectSequence_->valueMap());
    if (info && info->allDefaultNullBehavior && !info->notNull) {
      inNullPropagating_ = true;
      auto nullableInputs = gatherNullableInputs(expr);

      addInstruction<Nulls>(std::move(nullableInputs));
      enterNested();

      OperandIdx value = translateExpr(expr, result);

      inNullPropagating_ = false;
      leaveNested();
      addInstruction<NullsEnd>(value);
      return value;
    }
  }

  switch (expr->kind()) {
    case core::ExprKind::kFieldAccess:
    case core::ExprKind::kDereference: {
      std::vector<int32_t> path;
      if (!isField(expr, path)) {
        VELOX_NYI("field access to outside of input row");
      }
      auto it = stage_.fieldToOperand.find(expr.get());
      if (it != stage_.fieldToOperand.end()) {
        auto idx = it->second;
        if (!result.has_value()) {
          return idx;
        }
        if (result.has_value() && isOnlyUse(idx) && idx != result.value()) {
          Assignment* assignment = findAssignment(stage_.input, path);
          VELOX_CHECK_NOT_NULL(assignment);
          assignment->operand = result.value();
          return result.value();
        }
        if (result.has_value() && idx == result.value()) {
          return idx;
        }
        addInstruction<Assign>(result.value(), idx);
        return result.value();
      }
      VELOX_FAIL("Expect to have getters defined for : {}", expr->toString());
    }
    case core::ExprKind::kConstant: {
      auto idx = projectSequence_->makeConstant(expr);
      if (result.has_value() && result.value() != idx) {
        addInstruction<Assign>(result.value(), idx);
        return result.value();
      }
      return idx;
    }
    case core::ExprKind::kCall: {
      auto call = expr->asUnchecked<core::CallTypedExpr>();
      auto& name = call->name();
      auto special = specialForm(name);
      if (special) {
        auto specialResult = special(*this, call, result);
        if (specialResult != kNoOperand) {
          return specialResult;
        }
      }
      return makeCall(call->name(), expr->type(), call->inputs(), result);
    }
    case core::ExprKind::kCast:
      return makeCall("cast", expr->type(), expr->inputs(), result);
    default:
      VELOX_FAIL("Expr not supported: ", expr->toString());
  }
}

bool isField(const core::TypedExprPtr& expr, std::vector<int32_t>& path) {
  path.clear();

  auto current = expr;

  while (current) {
    if (auto fieldAccess = std::dynamic_pointer_cast<
            const facebook::velox::core::FieldAccessTypedExpr>(current)) {
      if (fieldAccess->inputs().empty()) {
        return fieldAccess->isInputColumn();
      }

      auto parent = fieldAccess->inputs()[0];
      if (parent->type()->isRow()) {
        auto fieldIndex =
            parent->type()->asRow().getChildIdx(fieldAccess->name());
        path.insert(path.begin(), fieldIndex);
        current = parent;
      } else {
        return false;
      }
    } else if (
        auto deref = std::dynamic_pointer_cast<
            const facebook::velox::core::DereferenceTypedExpr>(current)) {
      path.insert(path.begin(), deref->index());
      current = deref->inputs()[0];
    } else {
      return false;
    }
  }

  return false;
}

core::TypedExprPtr exprForPath(
    const core::AbstractProjectNode& project,
    const std::vector<int32_t>& path) {
  VELOX_CHECK(!path.empty(), "Path must have at least one element");

  core::TypedExprPtr expr = project.projections()[path[0]];

  for (size_t i = 1; i < path.size(); ++i) {
    expr = expr->inputs()[path[i]];
  }

  return expr;
}

FunctionLinearMetadata linearMetadata(const std::string& name) {
  auto& metadataMap = getLinearMetadataMap();
  auto it = metadataMap.find(name);
  if (it != metadataMap.end()) {
    return it->second;
  }
  // Return default metadata if not found.
  return FunctionLinearMetadata{};
}

void registerLinearMetadata(
    const std::string& name,
    FunctionLinearMetadata metadata) {
  auto& metadataMap = getLinearMetadataMap();
  metadataMap[name] = metadata;
}

TranslateSpecialForm specialForm(const std::string& name) {
  auto& specialFormMap = getSpecialFormMap();
  auto it = specialFormMap.find(name);
  if (it != specialFormMap.end()) {
    return it->second;
  }
  // Return nullptr if not found.
  return nullptr;
}

void registerSpecialForm(const std::string& name, TranslateSpecialForm form) {
  auto& specialFormMap = getSpecialFormMap();
  specialFormMap[name] = form;
}

void setupSpecialForms() {
  // Register "switch" special form
  registerSpecialForm(
      "switch",
      [](TranslateCtx& ctx,
         const core::CallTypedExpr* call,
         std::optional<OperandIdx> result) -> OperandIdx {
        std::vector<core::TypedExprPtr> inputs = call->inputs();
        return ctx.makeSwitch(call->type(), inputs, result);
      });

  // Register "if" special form
  registerSpecialForm(
      "if",
      [](TranslateCtx& ctx,
         const core::CallTypedExpr* call,
         std::optional<OperandIdx> result) -> OperandIdx {
        std::vector<core::TypedExprPtr> inputs = call->inputs();
        return ctx.makeSwitch(call->type(), inputs, result);
      });

  // Register "coalesce" special form
  registerSpecialForm(
      "coalesce",
      [](TranslateCtx& ctx,
         const core::CallTypedExpr* call,
         std::optional<OperandIdx> result) -> OperandIdx {
        const auto& inputs = call->inputs();
        VELOX_CHECK_EQ(inputs.size(), 2, "coalesce requires exactly 2 inputs");
        VELOX_CHECK(
            inputs[1]->kind() == core::ExprKind::kConstant,
            "coalesce second argument must be a constant");

        // Translate both inputs
        auto inputOperand = ctx.translateExpr(inputs[0], result);
        auto defaultOperand = ctx.translateExpr(inputs[1], std::nullopt);

        // Determine result operand
        OperandIdx resultOperand;
        if (result.has_value()) {
          resultOperand = result.value();
        } else {
          resultOperand = ctx.getTemp(inputs[0]->type());
        }

        // Add Coalesce instruction
        ctx.addInstruction<Coalesce>(
            resultOperand, inputOperand, defaultOperand);

        return resultOperand;
      });
}

namespace {

// Helper function to check if an expression is a constant with a specific bool
// value
bool isConstantBool(const core::TypedExprPtr& expr, bool value) {
  if (expr->kind() != core::ExprKind::kConstant) {
    return false;
  }
  auto constantExpr = expr->asUnchecked<core::ConstantTypedExpr>();
  if (constantExpr->type()->kind() != TypeKind::BOOLEAN) {
    return false;
  }
  return constantExpr->value().value<bool>() == value;
}

// Helper function to check if an expression is a constant null
bool isConstantNull(const core::TypedExprPtr& expr) {
  if (expr->kind() != core::ExprKind::kConstant) {
    return false;
  }
  auto constantExpr = expr->asUnchecked<core::ConstantTypedExpr>();
  return constantExpr->value().isNull();
}

// Helper function to create a constant bool expression
core::TypedExprPtr makeConstantBool(bool value) {
  return std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), variant(value));
}

// Helper function to create a constant null expression of a given type
core::TypedExprPtr makeConstantNull(const TypePtr& type) {
  return std::make_shared<core::ConstantTypedExpr>(
      type, variant::null(type->kind()));
}

} // namespace

void setupLinearMetadata() {
  // Register binary arithmetic functions that return the same type as
  // arguments. These functions can move an argument to the result.
  const std::vector<std::string> binaryArithmeticFunctions = {
      "plus",
      "minus",
      "multiply",
      "divide",
      "mod",
      "power",
      "bitwise_and",
      "bitwise_or",
      "bitwise_xor",
      "bitwise_left_shift",
      "bitwise_right_shift",
      "bitwise_arithmetic_shift_right"};

  FunctionLinearMetadata arithmeticMetadata;
  arithmeticMetadata.mayMoveArgToResult = true;

  for (const auto& funcName : binaryArithmeticFunctions) {
    registerLinearMetadata(funcName, arithmeticMetadata);
  }

  // Register "and" function
  FunctionLinearMetadata andMetadata;
  andMetadata.rewrite = [](const core::TypedExprPtr& expr,
                           ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    // If at least one argument is constant false, return constant false
    for (const auto& input : inputs) {
      if (isConstantBool(input, false)) {
        return makeConstantBool(false);
      }
    }

    // If at least one argument is constant null, return constant null
    for (const auto& input : inputs) {
      if (isConstantNull(input)) {
        return makeConstantNull(BOOLEAN());
      }
    }

    // If all arguments are constant true, return constant true
    bool allTrue = true;
    for (const auto& input : inputs) {
      if (!isConstantBool(input, true)) {
        allTrue = false;
        break;
      }
    }
    if (allTrue && !inputs.empty()) {
      return makeConstantBool(true);
    }

    return expr;
  };
  andMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                 ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    // If at least one argument is not notNull, return nullable
    for (const auto& input : inputs) {
      auto* info = valueInfo(input.get(), ctx.valueInfo);
      if (!info || !info->notNull) {
        return ValueInfo(false, false);
      }
    }
    return ValueInfo(true, true);
  };
  registerLinearMetadata("and", andMetadata);

  // Register "or" function
  FunctionLinearMetadata orMetadata;
  orMetadata.rewrite = [](const core::TypedExprPtr& expr,
                          ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    // If at least one argument is constant true, return constant true
    for (const auto& input : inputs) {
      if (isConstantBool(input, true)) {
        return makeConstantBool(true);
      }
    }

    // If at least one argument is constant null, return constant null
    for (const auto& input : inputs) {
      if (isConstantNull(input)) {
        return makeConstantNull(BOOLEAN());
      }
    }

    // If all arguments are constant false, return constant false
    bool allFalse = true;
    for (const auto& input : inputs) {
      if (!isConstantBool(input, false)) {
        allFalse = false;
        break;
      }
    }
    if (allFalse && !inputs.empty()) {
      return makeConstantBool(false);
    }

    return expr;
  };
  orMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    // If at least one argument is not notNull, return nullable
    for (const auto& input : inputs) {
      auto* info = valueInfo(input.get(), ctx.valueInfo);
      if (!info || !info->notNull) {
        return ValueInfo(false, false);
      }
    }
    return ValueInfo(true, true);
  };
  registerLinearMetadata("or", orMetadata);

  // Register "if" function
  FunctionLinearMetadata ifMetadata;
  ifMetadata.rewrite = [](const core::TypedExprPtr& expr,
                          ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (inputs.size() >= 3) {
      // If first argument is constant true, return second argument
      if (isConstantBool(inputs[0], true)) {
        return inputs[1];
      }
      // If first argument is constant false or constant null, return third
      // argument
      if (isConstantBool(inputs[0], false) || isConstantNull(inputs[0])) {
        return inputs[2];
      }
    }

    return expr;
  };
  ifMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (inputs.size() >= 3) {
      auto* thenInfo = valueInfo(inputs[1].get(), ctx.valueInfo);
      auto* elseInfo = valueInfo(inputs[2].get(), ctx.valueInfo);

      // If either second or third argument is nullable, return nullable
      bool notNull = true;
      if (!thenInfo || !thenInfo->notNull || !elseInfo || !elseInfo->notNull) {
        notNull = false;
      }
      return ValueInfo(notNull, notNull);
    }

    return ValueInfo(false, false);
  };
  registerLinearMetadata("if", ifMetadata);

  // Register "coalesce" function
  FunctionLinearMetadata coalesceMetadata;
  coalesceMetadata.rewrite = [](const core::TypedExprPtr& expr,
                                ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    VELOX_CHECK_EQ(inputs.size(), 2, "coalesce expects exactly 2 arguments");

    // If the first argument is non-nullable, return the first argument
    auto* firstInfo = valueInfo(inputs[0].get(), ctx.valueInfo);
    if (firstInfo && firstInfo->notNull) {
      return inputs[0];
    }

    // If the first argument is constant null, return the second argument
    if (isConstantNull(inputs[0])) {
      return inputs[1];
    }

    return expr;
  };
  coalesceMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                      ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (inputs.size() >= 2) {
      auto* firstInfo = valueInfo(inputs[0].get(), ctx.valueInfo);
      auto* secondInfo = valueInfo(inputs[1].get(), ctx.valueInfo);

      // If either first or second argument is not nullable, return not nullable
      bool notNull = false;
      if ((firstInfo && firstInfo->notNull) ||
          (secondInfo && secondInfo->notNull)) {
        notNull = true;
      }
      return ValueInfo(notNull, notNull);
    }

    return ValueInfo(false, false);
  };
  registerLinearMetadata("coalesce", coalesceMetadata);

  // Register "row_constructor" function
  FunctionLinearMetadata rowConstructorMetadata;
  rowConstructorMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                            ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    std::vector<ValueInfo> children;
    bool recursiveNotNull = true;

    for (const auto& input : inputs) {
      auto* inputInfo = valueInfo(input.get(), ctx.valueInfo);
      if (inputInfo) {
        children.push_back(*inputInfo);
        if (!inputInfo->recursiveNotNull) {
          recursiveNotNull = false;
        }
      } else {
        children.push_back(ValueInfo(false, false));
        recursiveNotNull = false;
      }
    }

    return ValueInfo(true, recursiveNotNull, std::move(children));
  };
  registerLinearMetadata("row_constructor", rowConstructorMetadata);

  // Register "isnull" function
  FunctionLinearMetadata isnullMetadata;
  isnullMetadata.rewrite = [](const core::TypedExprPtr& expr,
                              ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (!inputs.empty()) {
      auto* firstInfo = valueInfo(inputs[0].get(), ctx.valueInfo);
      // If the first argument is not nullable, return constant false
      if (firstInfo && firstInfo->notNull) {
        return makeConstantBool(false);
      }

      // If the first argument is constant null, return constant true
      if (isConstantNull(inputs[0])) {
        return makeConstantBool(true);
      }
    }

    return expr;
  };
  registerLinearMetadata("isnull", isnullMetadata);

  // Register "not" function
  FunctionLinearMetadata notMetadata;
  notMetadata.rewrite = [](const core::TypedExprPtr& expr,
                           ValueCtx& ctx) -> core::TypedExprPtr {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (!inputs.empty()) {
      // If the first argument is constant false, return constant true
      if (isConstantBool(inputs[0], false)) {
        return makeConstantBool(true);
      }

      // If the first argument is constant true, return constant false
      if (isConstantBool(inputs[0], true)) {
        return makeConstantBool(false);
      }

      // If the first argument is constant null, return constant null
      if (isConstantNull(inputs[0])) {
        return makeConstantNull(BOOLEAN());
      }
    }

    return expr;
  };
  notMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                 ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    if (!inputs.empty()) {
      auto* firstInfo = valueInfo(inputs[0].get(), ctx.valueInfo);
      // The ValueInfo is nullable if the first argument is nullable
      if (firstInfo && firstInfo->notNull) {
        return ValueInfo(true, true);
      }
    }

    return ValueInfo(false, false);
  };
  registerLinearMetadata("not", notMetadata);

  // Register "array_constructor" function
  FunctionLinearMetadata arrayConstructorMetadata;
  arrayConstructorMetadata.makeValueInfo = [](const core::TypedExprPtr& expr,
                                              ValueCtx& ctx) -> ValueInfo {
    auto call = expr->asUnchecked<core::CallTypedExpr>();
    const auto& inputs = call->inputs();

    std::vector<ValueInfo> children;
    bool recursiveNotNull = true;

    if (inputs.empty()) {
      // If the argument list is empty, create one child ValueInfo with
      // makeEmptyValueInfo
      auto arrayType = expr->type()->asArray();
      children.push_back(makeEmptyValueInfo(arrayType.elementType()));
    } else {
      // Initialize from the first argument's ValueInfo
      auto* firstInfo = valueInfo(inputs[0].get(), ctx.valueInfo);
      if (firstInfo) {
        children.push_back(*firstInfo);
        recursiveNotNull = firstInfo->recursiveNotNull;
      } else {
        children.push_back(ValueInfo(false, false));
        recursiveNotNull = false;
      }

      // Merge with ValueInfo of non-first arguments
      for (size_t i = 1; i < inputs.size(); ++i) {
        auto* inputInfo = valueInfo(inputs[i].get(), ctx.valueInfo);
        if (inputInfo) {
          mergeValueInfo(*inputInfo, children[0]);
        } else {
          ValueInfo tempInfo(false, false);
          mergeValueInfo(tempInfo, children[0]);
        }
      }

      recursiveNotNull = children[0].recursiveNotNull;
    }

    return ValueInfo(true, recursiveNotNull, std::move(children));
  };
  registerLinearMetadata("array_constructor", arrayConstructorMetadata);
}

} // namespace facebook::velox::exec
