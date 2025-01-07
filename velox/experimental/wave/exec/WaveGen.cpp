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

#include "velox/experimental/wave/exec/AggregateGen.h"
#include "velox/experimental/wave/exec/Project.h"
#include "velox/experimental/wave/exec/TableScan.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/Values.h"

namespace facebook::velox::wave {

thread_local int32_t CompileState::pipelineIdx_;
thread_local int32_t CompileState::kernelSeq_;
thread_local int32_t CompileState::branchIdx_;
thread_local PipelineCandidate* CompileState::currentCandidate_;
thread_local KernelBox* CompileState::currentBox_;

const std::string cudaTypeName(const Type& type) {
  switch (type.kind()) {
    case TypeKind::BIGINT:
      return "int64_t ";
    case TypeKind::BOOLEAN:
      return "bool ";
    default:
      VELOX_UNSUPPORTED("No gen for type {}", type.toString());
  }
}

const std::string cudaAtomicTypeName(const Type& type) {
  switch (type.kind()) {
    case TypeKind::BIGINT:
      return "long long";
    case TypeKind::INTEGER:
      return "int";
    default:
      VELOX_UNSUPPORTED(
          "Type not supported for Cuda atomic {}", type.toString());
  }
}

int32_t cudaTypeAlign(const Type& type) {
  return type.cppSizeInBytes();
}

int32_t cudaTypeSize(const Type& type) {
  return type.cppSizeInBytes();
}

bool KernelStep::references(AbstractOperand* op) {
  bool found = false;
  visitReferences([&](AbstractOperand* referenced) {
    if (found) {
      return;
    }
    if (op == referenced) {
      found = true;
    }
  });
  return found;
}

int32_t CompileState::ordinal(const AbstractOperand& op) {
  auto& params = selectedPipelines_[pipelineIdx_].levelParams[kernelSeq_];
  if (params.input.contains(op.id)) {
    return params.input.ordinal(op.id);
  }
  if (params.local.contains(op.id)) {
    return params.input.size() + params.local.ordinal(op.id);
  }
  if (params.output.contains(op.id)) {
    return params.input.size() + params.local.size() +
        params.output.ordinal(op.id);
  }
  VELOX_UNREACHABLE();
}

int32_t CompileState::stateOrdinal(const AbstractState& state) {
  auto& params = selectedPipelines_[pipelineIdx_].levelParams[kernelSeq_];
  return params.states.ordinal(state.id);
}

int32_t CompileState::declareVariable(const AbstractOperand& op) {
  auto ord = ordinal(op);
  if (declared_.contains(op.id)) {
    return ord;
  }
  declared_.add(op.id);
  declarations_ << fmt::format("{} r{};\n", cudaTypeName(*op.type), ord);
  return ord;
}

void CompileState::declareNamed(const std::string& line) {
  declarations_ << line << std::endl;
}

bool CompileState::hasMoreReferences(AbstractOperand* op, int32_t pc) {
  for (auto i = pc; i < currentBox_->steps.size(); ++i) {
    if (!currentBox_->steps[i]->preservesRegisters()) {
      return false;
    }
    if (currentBox_->steps[i]->references(op)) {
      return true;
    }
  }
  return false;
}

void CompileState::clearInRegister() {
  for (auto& op : operands_) {
    op->inRegister = false;
    op->registerNullBit = AbstractOperand::kNoNullBit;
  }
}

std::string KernelBox::toString() const {
  std::stringstream out;
  for (auto i = 0; i < steps.size(); ++i) {
    out << i << ": " << steps[i]->toString();
  }
  return out.str();
}

void NullCheck::visitReferences(
    std::function<void(AbstractOperand*)> visitor) const {
  for (auto& op : operands) {
    visitor(op);
  }
}

void NullCheck::generateMain(CompileState& state, int32_t /*syncLable*/) {
  std::vector<AbstractOperand*> lastUse;
  bool isFirst = true;
  state.setInsideNullPropagating(true);

  for (auto* op : operands) {
    if (!op->inRegister && state.hasMoreReferences(op, endIdx + 1)) {
      if (isFirst) {
        state.declareNamed(fmt::format("bool anyNull{};", label));
        state.generated() << fmt::format("  anyNull{} = false;\n", label);
        isFirst = false;
      }
      auto& flags = state.flags(*op);
      bool mayWrap = state.mayWrap(flags.wrappedAt);
      auto ordinal = state.declareVariable(*op);
      state.generated() << fmt::format(
          "anyNull{} |= setRegisterNull(nulls{}, {}, !valueOrNull<{}>(operands, {}, blockBase, r{}));\n",
          label,
          ordinal / 32,
          ordinal & 31,
          mayWrap ? "true" : "false",
          ordinal,
          ordinal);
      op->inRegister = true;
    } else {
      lastUse.push_back(op);
    }
  }
  if (!isFirst) {
    state.generated() << fmt::format(
        "if (anyNull{}) {{ goto end{};}}\n", label, label);
  }
  for (auto* op : lastUse) {
    if (op->inRegister) {
      auto ord = state.ordinal(*op);
      state.generated() << fmt::format(
          "if (isRegisterNull(nulls{}, {})) {{goto end{};}}\n",
          ord / 32,
          ord & 31,
          label);
      continue;
    }
    auto& flags = state.flags(*op);

    bool mayWrap = state.mayWrap(flags.wrappedAt);
    auto ord = state.declareVariable(*op);
    state.generated() << fmt::format(
        "if (!valueOrNull<{}>(operands, {}, blockBase, r{})) {{goto end{};}}\n",
        mayWrap ? "true" : "false",
        ord,
        ord,
        label);
    op->inRegister = true;
  }
}

std::string NullCheck::toString() const {
  std::stringstream out;
  out << "NullCheck: null in (";
  for (auto& op : operands) {
    out << op->toString() << " ";
  }
  out << ") makes null in " << result->toString() << std::endl;
  return out.str();
}

void EndNullCheck::generateMain(CompileState& state, int32_t /*syncLable*/) {
  auto ord = state.ordinal(*result);
  state.generated() << fmt::format("goto skip{};\n", label)
                    << fmt::format("end{}: \n", label);
  auto flags = state.flags(*result);
  state.generated() << fmt::format(
      "setRegisterNull(nulls{}, {}, true);\n", ord / 32, ord & 31, true);
  if (flags.needStore) {
    state.generated() << fmt::format(
        "setNull(operands, {}, blockBase, true);\n", ord);
  }
  state.generated() << fmt::format("skip{}: ;\n", label);
  state.setInsideNullPropagating(false);
}

std::string CompileState::literalText(const AbstractOperand& op) {
  auto& constant = op.constant;
  switch (op.type->kind()) {
    case TypeKind::BIGINT:
      return fmt::format(
          "{}LL", constant->as<SimpleVector<int64_t>>()->valueAt(0));
    case TypeKind::INTEGER:
      return fmt::format(
          "{} ", constant->as<SimpleVector<int32_t>>()->valueAt(0));
    case TypeKind::SMALLINT:
      return fmt::format(
          "{} ", constant->as<SimpleVector<int16_t>>()->valueAt(0));
    case TypeKind::TINYINT:
      return fmt::format(
          "{} ", constant->as<SimpleVector<int8_t>>()->valueAt(0));
    case TypeKind::REAL:
      return fmt::format(
          "{} ", constant->as<SimpleVector<float>>()->valueAt(0));
    case TypeKind::DOUBLE:
      return fmt::format(
          "{} ", constant->as<SimpleVector<double>>()->valueAt(0));
    default:
      VELOX_NYI("Unsupported type");
  }
}

void CompileState::generateOperand(const AbstractOperand& op) {
  if (op.constant) {
    generated_ << literalText(op);
    return;
  }
  if (op.inRegister && insideNullPropagating_) {
    generated_ << fmt::format(" r{} ", ordinal(op));
    return;
  }
  if (op.notNull || insideNullPropagating_) {
    auto& flags = this->flags(op);
    bool mayWrap = this->mayWrap(flags.wrappedAt);
    generated_ << fmt::format(
        "nonNullOperand<{}, {}>(operands, {}, blockBase)",
        cudaTypeName(*op.type),
        mayWrap,
        ordinal(op));
  }
}

void Compute::generateMain(CompileState& state, int32_t /*syncLable*/) {
  VELOX_CHECK_NOT_NULL(operand->expr);
  auto& flags = state.flags(*operand);
  auto ord = state.declareVariable(*operand);
  state.functionReferenced(operand);
  state.generated() << fmt::format("r{} = {}(", ord, operand->expr->name());
  for (auto i = 0; i < operand->inputs.size(); ++i) {
    state.generateOperand(*operand->inputs[i]);
    if (i < operand->inputs.size() - 1) {
      state.generated() << ", ";
    }
  }
  state.generated() << ");\n";
  operand->inRegister = true;
  if (flags.needStore) {
    operand->isStored = true;
    state.generated() << fmt::format(
        "flatResult<{}>(operands, {}, blockBase) = r{};\n",
        cudaTypeName(*operand->type),
        ord,
        ord);
  }
}

std::string Compute::toString() const {
  std::stringstream out;
  if (operand->expr) {
    out << operand->expr->name() << "(";
    for (auto& in : operand->inputs) {
      out << in->toString() << " ";
    }
    out << ")" << std::endl;
  } else {
    out << operand->toString() << std::endl;
  }
  return out.str();
}

void CompileState::ensureOperand(AbstractOperand* op) {
  if (op->inRegister) {
    return;
  }
  auto& flags = this->flags(*op);
  bool mayWrap = this->mayWrap(flags.wrappedAt);
  if (op->isStored) {
    auto ord = declareVariable(*op);
    if (op->notNull) {
      generated_ << fmt::format(
          "  r{} = nonNullOperand<{}>(operands, {}, blockBase);\n",
          mayWrap,
          ord,
          ord);
    } else {
      generated_ << fmt::format(
          "  loadValueOrNull<{}>(operands, {}, blockBase, r{}, nulls{});\n",
          mayWrap,
          ord,
          ord,
          ord / 32);
    }
    op->inRegister = true;
  } else {
    VELOX_FAIL("Expression should have been generated at this point.");
  }
}

std::string CompileState::isNull(const AbstractOperand* op) {
  auto ord = ordinal(*op);
  if (op->notNull) {
    return "false";
  }
  if (op->inRegister) {
    return fmt::format("(0 == (nulls{} & (1U << {})))", ord / 32, ord & 31);
  }
  VELOX_FAIL("Expecting op in register");
}

std::string CompileState::operandValue(const AbstractOperand* op) {
  VELOX_CHECK(op->inRegister);
  return fmt::format("r{}", ordinal(*op));
}

std::string CompileState::generateIsTrue(const AbstractOperand& op) {
  auto ord = ordinal(op);
  if (op.inRegister) {
    if (op.notNull) {
      generated_ << fmt::format("bool flag{} = r{}", ord, ord);
    } else {
      generated_ << fmt::format(
          "bool flag{} = r{} && !isRegisterNull(nulls{}, {});\n",
          ord,
          ord,
          ord / 32,
          ord & 31);
    }
  } else {
    auto& flags = this->flags(op);
    bool mayWrap = this->mayWrap(!flags.wrappedAt);
    if (op.notNull || insideNullPropagating_) {
      generated_ << fmt::format(
          "bool flag{} = nonNullOperand<bool, {}>(operands, {}, blockBase)",
          ord,
          mayWrap,
          ord);
    } else {
      generated_ << fmt::format("bool flag{};\n", ord);
      generated_ << fmt::format(
          "if (!valueOrNull<{}, bool>(operands, {}, blockBase, flags{})) {{ flags{} = false; }};\n",
          mayWrap ? "true" : "false",
          ord,
          ord,
          ord);
    }
  }
  return fmt::format("flag{}", ord);
}

void CompileState::functionReferenced(const AbstractOperand* op) {
  auto numInput = op->inputs.size();
  std::vector<TypePtr> types;
  types.reserve(numInput);
  for (auto i = 0; i < numInput; ++i) {
    types.push_back(op->expr->inputs()[i]->type());
  }
  functionReferenced(op->expr->name(), types, op->type);
}

void CompileState::addInclude(const std::string& path) {
  auto line = fmt::format("#include \"{}\"", path);
  if (includes_.count(line) != 0) {
    return;
  }
  includes_.insert(line);
  includeText_ << line << std::endl;
}

void CompileState::functionReferenced(
    const std::string& name,
    const std::vector<TypePtr>& types,
    const TypePtr& resultType) {
  FunctionKey key(name, types);

  if (functions_.count(key)) {
    return;
  }
  functions_.insert(key);
  auto definition = waveRegistry().makeDefinition(key, resultType);
  if (!definition.includeLine.empty() &&
      includes_.find(definition.includeLine) == includes_.end()) {
    includes_.insert(definition.includeLine);
    includeText_ << definition.includeLine << std::endl;
  }
  inlines_ << "inline __device__ " << definition.definition << std::endl;
}

int32_t CompileState::nextWrapId() {
  return ++wrapId_;
}

int32_t CompileState::wrapLiteral(int32_t nthWrap) {
  // We take one Operand of each group of Operands that shares a wrappedAt such
  // that the Operand's lifetime crosses the filter.
  CodePosition filter(kernelSeq_, 0, stepIdx_);
  std::unordered_set<int32_t> wraps;
  std::vector<OperandIndex> ordinals;
  for (auto& op : operands_) {
    auto& flags = currentCandidate_->flags(op.get());
    if (!flags.lastUse.empty() && !flags.definedIn.empty() &&
        filter.isBefore(flags.lastUse) && flags.definedIn.isBefore(filter)) {
      auto wrappedAt = flags.wrappedAt;
      if (wrappedAt == AbstractOperand::kNoWrap) {
        op->wrappedAt = nthWrap;
        flags.wrappedAt = nthWrap;
        wrappedAt = nthWrap;
      }
      if (wraps.count(wrappedAt)) {
        continue;
      }
      wraps.insert(wrappedAt);
      ordinals.push_back(ordinal(*op));
    }
  }
  generated_ << fmt::format("const OperandIndex wraps{}[] = {{", nthWrap);
  for (auto i = 0; i < ordinals.size(); ++i) {
    generated_ << ordinals[i];
    if (i < ordinals.size() - 1) {
      generated_ << ", ";
    }
  }
  generated_ << "};\n";
  return ordinals.size();
}

void Filter::generateMain(CompileState& state, int32_t syncLabel) {
  auto flagValue = state.generateIsTrue(*flag);
  auto& out = state.generated();
  out << fmt::format(" sync{}:\n", syncLabel);
  out << fmt::format(
      "filterKernel({}, operands, {}, blockBase, shared, laneStatus);\n",
      flagValue,
      state.ordinal(*indices));
  auto numWraps = state.wrapLiteral(nthWrap);
  out << fmt::format(
      "wrapKernel(wraps{}, {}, {}, operands, blockBase, shared);\n",
      nthWrap,
      numWraps,
      state.ordinal(*indices));
  state.lastPlacedWrap() = nthWrap;
  state.clearInRegister();
}

void AggregateProbe::generateMain(CompileState& state, int32_t syncLabel) {
  makeAggregateOps(state, *this, false);
  makeAggregateProbe(state, *this, syncLabel);
}

std::string AggregateProbe::preContinueCode(CompileState& state) {
  return "    laneStatus = laneStatus == ErrorCode::kInsufficientMemory\n"
         "      ? ErrorCode::kOk : ErrorCode::kInactive;\n";
}

std::unique_ptr<AbstractInstruction> AggregateProbe::addInstruction(
    CompileState& state) {
  RowTypePtr type;
  static std::vector<AbstractAggInstruction> empty;
  auto agg = std::make_unique<AbstractAggregation>(
      state.nextSerial(), keys, empty, this->state, type);
  int32_t offset =
      sizeof(int32_t) + bits::roundUp(keys.size() + updates.size(), 32) / 8;
  for (auto& key : keys) {
    int32_t align = cudaTypeAlign(*key->type);
    int32_t width = cudaTypeSize(*key->type);
    offset = bits::roundUp(offset, align) + width;
  }
  for (auto& update : updates) {
    auto [size, align] = update->generator->accumulatorSizeAndAlign(*update);
    offset = bits::roundUp(offset, align) + size;
  }
  agg->roundedRowSize = bits::roundUp(offset, 8);
  abstractAggregation = agg.get();
  return agg;
}

void AggregateUpdate::generateMain(CompileState& state, int32_t /*syncLabel*/) {
}

void ReadAggregation::generateMain(CompileState& state, int32_t /*syncLabel*/) {
  visitResults([&](auto op) { op->isStored = true; });
  makeAggregateOps(state, *probe, true);
  makeReadAggregation(state, *this);
}

std::unique_ptr<AbstractInstruction> ReadAggregation::addInstruction(
    CompileState& state) {
  return std::make_unique<AbstractReadAggregation>(
      state.nextSerial(), probe->abstractAggregation);
}

void writeDebugFile(const KernelSpec& spec) {
  try {
    std::ofstream out(spec.filePath, std::ios_base::out | std::ios_base::trunc);
    out << spec.code;
    out.close();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error saving compiled file /tmp/" << spec.filePath << " "
               << e.what();
  }
}

void CompileState::generateSkip() {
  generated_ << fmt::format(
      "  if (laneStatus != ErrorCode::kOk) {{ goto sync{}; }}\n",
      nextSyncLabel_);
}

int32_t findLastWrap(const PipelineCandidate& candidate, int32_t kernelSeq) {
  for (int32_t k = kernelSeq - 1; k >= 0; --k) {
    if (candidate.steps[k].size() > 1) {
      continue;
    }
    auto& steps = candidate.steps[k][0].steps;
    for (int32_t i = steps.size() - 1; i >= 0; --i) {
      auto s = steps[i]->isWrap();
      if (s != AbstractOperand::kNoWrap) {
        return s;
      }
    }
  }
  return -1;
}

std::string checkLaneStatus() {
#ifdef BLOCK_STATUS_CHECK
  return "  if ((int)laneStatus > 4) {\n"
         "printf(\"bad laneStatus\\n\");\n"
         "  }\n";
#else
  return "";
#endif
}

ProgramKey CompileState::makeLevelText(
    int32_t pipelineIdx,
    int32_t kernelSeq,
    KernelSpec& spec) {
  std::lock_guard<std::mutex> l(generateMutex_);
  VELOX_CHECK(generated_.str().empty());
  VELOX_CHECK(inlines_.str().empty());
  sharedSize_ = 0;
  insideNullPropagating_ = false;
  declared_ = OperandSet();
  currentCandidate_ = &selectedPipelines_[pipelineIdx];
  pipelineIdx_ = pipelineIdx;
  kernelSeq_ = kernelSeq;
  auto& level = selectedPipelines_[pipelineIdx_].steps[kernelSeq_];
  lastPlacedWrap_ = findLastWrap(*currentCandidate_, kernelSeq);

  VELOX_CHECK_EQ(1, level.size(), "Only one program per level supported");
  std::stringstream head;
  auto kernelName = fmt::format("wavegen{}", ++kernelCounter_);
  kernelEntryPoints_ = {fmt::format("facebook::velox::wave::{}", kernelName)};
  generated_ << "  GENERATED_PREAMBLE(0);\n";
  for (branchIdx_ = 0; branchIdx_ < level.size(); ++branchIdx_) {
    auto& box = level[branchIdx_];
    currentBox_ = &box;
    clearInRegister();
    bool anyRetry = false;
    bool needActiveCheck = true;
    generated_ << "if (!shared->isContinue) {\n"
               << checkLaneStatus() << "  }\n";

    for (stepIdx_ = 0; stepIdx_ < box.steps.size(); ++stepIdx_) {
      auto* step = box.steps[stepIdx_];
      auto label = step->continueLabel();
      if (label.has_value()) {
        if (!anyRetry) {
          anyRetry = true;
          generated_ << "if (shared->isContinue) {\n"
                     << checkLaneStatus() << "switch(shared->startLabel) {\n";
        }
        generated_ << fmt::format(
            "case {}: {} goto continue{};\n",
            label.value(),
            step->preContinueCode(*this),
            label.value());
      }
    }
    if (anyRetry) {
      generated_ << "    case 0xffff: return;\n}\n}\n";
    }
    for (stepIdx_ = 0; stepIdx_ < box.steps.size(); ++stepIdx_) {
      if (needActiveCheck) {
        for (auto next = stepIdx_; next < box.steps.size(); ++next) {
          auto label = box.steps[next]->continueLabel();
          if (label.has_value()) {
            generated_ << fmt::format(" continue{}:\n", label.value());
          }
          if (box.steps[next]->isBarrier()) {
            break;
          }
        }
        generateSkip();
        needActiveCheck = false;
      }
      // Generate the  code for first execution.
      auto step = box.steps[stepIdx_];
      sharedSize_ = std::max<int32_t>(sharedSize_, step->sharedMemorySize());

      int32_t syncLabel = -1;
      if (step->isBarrier()) {
        syncLabel = nextSyncLabel_;
        ++nextSyncLabel_;
        needActiveCheck = true;
      }
      step->generateMain(*this, syncLabel);
    }
  }
  generated_ << fmt::format("sync{}: ;\n", nextSyncLabel_);
  generated_ << " PROGRAM_EPILOGUE();\n}\n}\n";
  head
      << "#include \"velox/experimental/wave/exec/WaveCore.cuh\"\n"
      << includeText_.str() << std::endl
      << "namespace facebook::velox::wave {\n"
      << inlines_.str() << std::endl
      << fmt::format(
             "void __global__ __launch_bounds__(1024) {}(KernelParams params) {{\n",
             kernelName);

  auto& params = currentCandidate_->levelParams[kernelSeq_];
  int32_t numRegs =
      params.input.size() + params.local.size() + params.output.size();
  for (auto i = 0; i < numRegs; i += 32) {
    head << fmt::format(" uint32_t nulls{} = ~0;\n", i / 32);
  }
  head << declarations_.str();
  head << generated_.str();

  // Reset the generated text and state before generating the next kernel.
  generated_ = std::stringstream();
  declarations_ = std::stringstream();
  inlines_ = std::stringstream();
  includeText_ = std::stringstream();
  includes_.clear();
  functions_.clear();
  std::vector<AbstractOperand*> input;
  std::vector<AbstractOperand*> local;
  std::vector<AbstractOperand*> output;
  params.input.forEach(
      [&](int32_t id) { input.push_back(operands_[id].get()); });

  params.local.forEach(
      [&](int32_t id) { local.push_back(operands_[id].get()); });
  params.output.forEach(
      [&](int32_t id) { output.push_back(operands_[id].get()); });

  spec.code = head.str();
  spec.entryPoints = std::move(kernelEntryPoints_);
  spec.filePath = fmt::format("/tmp/{}.cu", kernelName);
  // Write the geneerated code to a file for debugger.
  writeDebugFile(spec);

  return ProgramKey{
      head.str(), std::move(input), std::move(local), std::move(output)};
}

bool CompileState::isWrapInParams(int32_t nthWrap, const LevelParams& params) {
  bool found = false;
  params.input.forEach(
      [&](int32_t id) { found |= operands_[id]->wrappedAt == nthWrap; });
  return found;
}

void CompileState::fillExtraWrap(OperandSet& extraWrap) {
  auto& candidate = selectedPipelines_[pipelineIdx_];
  // Loop over all operands in the pipeline. If there are wraps in the current
  // level, mark the operands not in the kernel but defined before and accessed
  // after as extr raps.
  if (candidate.steps[kernelSeq_].size() > 1) {
    // If there are multiple branches there is no cardinality change or wraps
    // from the level.
    return;
  }
  auto& box = candidate.steps[kernelSeq_][0];
  int32_t nthWrap = AbstractOperand::kNoWrap;
  for (auto& step : box.steps) {
    nthWrap = step->isWrap();
    if (nthWrap != AbstractOperand::kNoWrap) {
      break;
    }
  }
  if (nthWrap == AbstractOperand::kNoWrap) {
    return;
  }
  auto params = candidate.levelParams[kernelSeq_];
  OperandSet wraps;
  params.input.forEach([&](int32_t id) {
    auto* op = operands_[id].get();
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      wraps.add(op->wrappedAt);
    }
  });
  for (auto i = 0; i < candidate.operandFlags.size(); ++i) {
    auto& flags = candidate.operandFlags[i];
    if (flags.definedIn.empty() || params.input.contains(i)) {
      continue;
    }
    if (flags.definedIn.kernelSeq < kernelSeq_ &&
        flags.lastUse.kernelSeq > kernelSeq_ &&
        (operands_[i]->wrappedAt == AbstractOperand::kNoWrap ||
         operands_[i]->wrappedAt > nthWrap)) {
      operands_[i]->wrappedAt = nthWrap;
      // We need to add the wrap to extra wraps if no existing parameter of the
      // kernel has the wrap.
      if (!wraps.contains(nthWrap)) {
        extraWrap.add(nthWrap);
      }
    }
  }
}

void CompileState::makeLevel(std::vector<KernelBox>& level) {
  VELOX_CHECK_EQ(1, level.size(), "Only one program per level supported");
  std::vector<std::unique_ptr<AbstractInstruction>> instructions;
  int32_t kernelEntryPointCounter = 1;
  std::unordered_map<int32_t, int32_t> kernelEntryPoints;
  for (branchIdx_ = 0; branchIdx_ < level.size(); ++branchIdx_) {
    currentBox_ = &level[branchIdx_];
    for (stepIdx_ = 0; stepIdx_ < currentBox_->steps.size(); ++stepIdx_) {
      auto instructionUnique =
          currentBox_->steps[stepIdx_]->addInstruction(*this);
      if (instructionUnique) {
        instructions.push_back(std::move(instructionUnique));
        auto* instruction = instructions.back().get();
        instruction->reserveState(instructionStatus_);
        auto* status = instruction->mutableInstructionStatus();
        if (status) {
          currentBox_->steps[stepIdx_]->status = *status;
        }
        auto opInst = dynamic_cast<AbstractOperator*>(instruction);
        if (opInst) {
          if (auto* agg = dynamic_cast<AbstractAggregation*>(opInst)) {
            kernelEntryPoints[agg->continueIdx()] = kernelEntryPointCounter++;
          }
          AbstractState* state = opInst->state;
          state->instruction = instruction;
        }
      }
    }
  }

  programs_.clear();
  KernelSpec spec;
  makeLevelText(pipelineIdx_, kernelSeq_, spec);
  auto kernel = CompiledKernel::getKernel(spec.code, [spec]() { return spec; });
  // Sync with compilation to serialize compile order.
  kernel->info(0);

  auto& params = currentCandidate_->levelParams[kernelSeq_];
  auto numBranches = currentCandidate_->steps[kernelSeq_].size();
  OperandSet extraWrap;
  fillExtraWrap(extraWrap);
  std::vector<std::unique_ptr<ProgramState>> states;
  params.states.forEach([&](int32_t id) {
    auto* abstractState = operatorStates_[id].get();
    auto programState = std::make_unique<ProgramState>();
    programState->stateId = abstractState->id;
    auto* abstractInst =
        reinterpret_cast<AbstractAggregation*>(abstractState->instruction);
    programState->isGlobal = true;
    programState->create =
        [inst = abstractInst](
            WaveStream& stream) -> std::shared_ptr<OperatorState> {
      auto newState = std::make_shared<AggregateOperatorState>();
      newState->instruction = inst;
      stream.makeAggregate(*inst, *newState);
      return newState;
    };

    states.push_back(std::move(programState));
  });
  auto program = std::make_shared<Program>(
      params.input,
      params.local,
      params.output,
      extraWrap,
      numBranches,
      sharedSize_,
      operands_,
      std::move(states),
      std::move(kernel));
  for (auto& pair : kernelEntryPoints) {
    program->addEntryPointForSerial(pair.first, pair.second);
  }
  for (auto& i : instructions) {
    program->add(std::move(i));
  }
  programs_.push_back(std::move(program));
}

bool emptyLevel(std::vector<KernelBox> level) {
  return level.empty() || level[0].steps.empty();
}

// Sets 'op's output operand ids to be the outputs generated by steps from
// 'begin' to 'end'.
void PipelineCandidate::setOutputIds(
    CompileState* state,
    WaveOperator* op,
    int32_t begin,
    int32_t end) {
  for (auto i = begin; i < end; ++i) {
    auto& params = levelParams[i];
    params.output.forEach([&](auto id) {
      op->addOutputId(id);
      auto* operand = state->operandById(id);
      operand->isStored = true;
    });
  }
}

void CompileState::setOperandByCandidate(PipelineCandidate& candidate) {
  for (auto i = 0; i < candidate.operandFlags.size() && i < operands_.size();
       ++i) {
    auto& flags = candidate.operandFlags[i];
    if (!flags.definedIn.empty() &&
        flags.wrappedAt != AbstractOperand::kNoWrap) {
      operands_[i]->wrappedAt = flags.wrappedAt;
    }
  }
}

void CompileState::generatePrograms() {
  // We can move the per-candidate Operand flags to the operands themselves.
  for (pipelineIdx_ = 0; pipelineIdx_ < selectedPipelines_.size();
       ++pipelineIdx_) {
    setOperandByCandidate(selectedPipelines_[pipelineIdx_]);
  }
  for (pipelineIdx_ = 0; pipelineIdx_ < selectedPipelines_.size();
       ++pipelineIdx_) {
    currentCandidate_ = &selectedPipelines_[pipelineIdx_];
    auto& firstStep = currentCandidate_->steps[0][0].steps.front();
    int32_t start = 0;
    if (firstStep->kind() == StepKind::kTableScan) {
      auto& scanStep = firstStep->as<TableScanStep>();
      operators_.push_back(std::make_unique<TableScan>(
          *this,
          operators_.size(),
          *scanStep.node,
          std::move(scanStep.defines)));
      start = 1;
    }
    if (firstStep->kind() == StepKind::kValues) {
      operators_.push_back(
          std::make_unique<Values>(*this, *firstStep->as<ValuesStep>().node));
      start = 1;
    }

    if (start == 1) {
      currentCandidate_->setOutputIds(this, operators_[0].get(), 0, 1);
    }
    for (kernelSeq_ = start; kernelSeq_ < currentCandidate_->steps.size();
         ++kernelSeq_) {
      if (emptyLevel(currentCandidate_->steps[kernelSeq_])) {
        continue;
      }
      makeLevel(currentCandidate_->steps[kernelSeq_]);
    }
    std::vector<std::vector<ProgramPtr>> levels;
    for (auto& program : programs_) {
      levels.emplace_back();
      levels.back().push_back(std::move(program));
    }
    if (levels.empty()) {
      return;
    }
    operators_.push_back(std::make_unique<Project>(
        *this,
        selectedPipelines_[pipelineIdx_].outputType,
        std::move(levels),
        nullptr));
    currentCandidate_->setOutputIds(
        this, operators_.back().get(), start, currentCandidate_->steps.size());
  }
}

} // namespace facebook::velox::wave
