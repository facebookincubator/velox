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

#include "velox/experimental/torchwave/DeviceSymbols.h"

#include <fmt/format.h>
#include <folly/synchronization/CallOnce.h>
#include <glog/logging.h>
#include <cstddef>
#include <functional>

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/StaticDims.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

namespace {

// The fixup kernel writes these fields of a parameter Tensor by offset.
static_assert(offsetof(Tensor, storage) == 0);
static_assert(offsetof(Tensor, elementSize) == 9);
static_assert(offsetof(Tensor, dims) == 12);
static_assert(offsetof(Tensor, strides) == 28);
static_assert(offsetof(Tensor, numEl) == 44);
static_assert(offsetof(Tensor, status) == 48);
static_assert(offsetof(Tensor, contiguous) == 52);
static_assert(kMaxDims == 4);

// Depth of the operand stack of evalProgram in the fixup kernel source.
constexpr int32_t kProgramStackDepth = 16;
constexpr int32_t kRecordWords = 12;
constexpr int32_t kTensorRecord = 0;
constexpr int32_t kScalarRecord = 1;
// A tensor parameter that carries a shape and no data: dims and numEl only.
constexpr int32_t kShapeRecord = 2;
// A concat band whose shape the host knows: only its start moves.
constexpr int32_t kBandRecord = 3;
// A tensor record for a slice the host views from the start of its base: the
// start moves from 0 rather than from its value over the capacities.
constexpr int32_t kViewRecord = 4;
// Dim and stride codes in a tensor record that are not program offsets.
constexpr int32_t kHostValue = -1;
constexpr int32_t kDenseStride = -2;

// Program words: the opcode in the top byte, an operand below it.
enum Opcode : int64_t {
  kPushConst = 1,
  kPushSymbol = 2,
  kAdd = 3,
  kMul = 4,
  kMax = 5,
  kMin = 6,
  kPow = 7,
  kFloorDiv = 8,
  kMod = 9,
  kEnd = 10,
};

constexpr int64_t kOperandBits = 56;
constexpr int64_t kOperandMask = (int64_t{1} << kOperandBits) - 1;

int64_t word(Opcode opcode, int64_t operand = 0) {
  return (static_cast<int64_t>(opcode) << kOperandBits) |
      (operand & kOperandMask);
}

const char* const kFixupSource = R"(
constexpr int kMaxSymbols = 1024;

// TW_SETUP_CLOCKS marks; see DeviceGrid.cpp.
__shared__ long long setupStart;
__shared__ int fixupMark[3];
#ifdef TW_SETUP_CLOCKS
#define FIXUP_MARK(k)                                                 \
  if (tid == 0) {                                                     \
    fixupMark[k] = static_cast<int>(clock64() - setupStart);          \
  }
#else
#define FIXUP_MARK(k)
#endif
constexpr int kSharedCode = 2048;
constexpr int kDenseStride = -2;

__device__ inline long long operandOf(long long word) {
  long long operand = word & ((1LL << 56) - 1);
  return (operand & (1LL << 55)) ? operand - (1LL << 56) : operand;
}

__device__ long long evalProgram(const long long* code, const long long* values) {
  // Most programs are one constant or symbol; those skip the stack, which
  // lives in local memory.
  const long long first = code[0];
  if ((code[1] >> 56) == 10) {
    const long long op = first >> 56;
    if (op == 1) {
      return operandOf(first);
    }
    if (op == 2) {
      return values[operandOf(first)];
    }
  }
  long long stack[16];
  int top = 0;
  for (;; ++code) {
    const long long op = *code >> 56;
    long long operand = *code & ((1LL << 56) - 1);
    if (operand & (1LL << 55)) {
      operand -= (1LL << 56);
    }
    if (op == 10) {
      return top > 0 ? stack[top - 1] : 0;
    }
    if (op == 1) {
      stack[top++] = operand;
      continue;
    }
    if (op == 2) {
      stack[top++] = values[operand];
      continue;
    }
    const long long b = stack[--top];
    const long long a = stack[--top];
    long long r = 0;
    switch (op) {
      case 3: r = a + b; break;
      case 4: r = a * b; break;
      case 5: r = a > b ? a : b; break;
      case 6: r = a < b ? a : b; break;
      case 7: { r = 1; for (long long i = 0; i < b; ++i) r *= a; break; }
      case 8: { r = b == 0 ? 0 : a / b; if (b != 0 && (a % b != 0) && ((a < 0) != (b < 0))) --r; break; }
      case 9: { long long q = b == 0 ? 0 : a / b; if (b != 0 && (a % b != 0) && ((a < 0) != (b < 0))) --q; r = a - q * b; break; }
    }
    stack[top++] = r;
  }
}

// Waits for 'count' threads at named barrier 'id'; 0 is __syncthreads.
__device__ inline void barrier(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// barrier() that also returns whether 'predicate' is nonzero in any thread.
__device__ inline int barrierOr(int id, int count, int predicate) {
  int result;
  asm volatile(
      "{ .reg .pred p, q; setp.ne.s32 p, %1, 0; "
      "bar.red.or.pred q, %2, %3, p; selp.s32 %0, 1, 0, q; }"
      : "=r"(result)
      : "r"(predicate), "r"(id), "r"(count)
      : "memory");
  return result;
}

// Run by 'numThreads' threads from 'firstThread' on, which meet at named
// barrier 'barrierId'; the rest of the block may do other work. The programs
// go to 'codeBuffer' in shared memory when it is given.
__device__ void fixupBody(
    unsigned char* params,
    const int* records,
    const long long* code,
    int numRecords,
    const long long* symbols,
    int numSymbols,
    long long* codeBuffer,
    int numCode,
    int firstThread,
    int numThreads,
    int barrierId) {
  const int tid = threadIdx.x - firstThread;
  // 'symbols' holds numSymbols source addresses, then their capacities.
  __shared__ long long values[kMaxSymbols];
  __shared__ long long capacities[kMaxSymbols];
  // All loads go before the one barrier: the programs, the symbols, and a
  // prefetch of each thread's first record's slot. Each word of a program in
  // global memory is otherwise a dependent load.
  const int firstRecord = tid < numRecords ? records[tid * 12] : -1;
  if (codeBuffer != nullptr) {
    for (int i = tid; i < numCode; i += numThreads) {
      codeBuffer[i] = code[i];
    }
    code = codeBuffer;
  }
  int anyOver = 0;
  for (int i = tid; i < numSymbols; i += numThreads) {
    const long long address = symbols[i];
    const long long capacity = symbols[numSymbols + i];
    const long long v = address == 0 ? 0
        : (address & 1) ? *reinterpret_cast<const int*>(address - 1)
                        : *reinterpret_cast<const long long*>(address);
    values[i] = v;
    capacities[i] = capacity;
    anyOver |= v < 0 || v > capacity ? 1 : 0;
  }
  if (firstRecord >= 0) {
    asm volatile("prefetch.global.L1 [%0];" ::"l"(params + firstRecord));
  }
  FIXUP_MARK(0);
  const int over = barrierOr(barrierId, numThreads, anyOver);
  FIXUP_MARK(1);
  for (int r = tid; r < numRecords; r += numThreads) {
    const int* rec = records + r * 12;
    unsigned char* slot = params + rec[0];
    if (rec[1] == 1) {
      *reinterpret_cast<long long*>(slot) = over ? 0 : evalProgram(code + rec[4], values);
      continue;
    }
    // A band the host placed at its start over the capacities moves to its
    // start over the real values.
    if (rec[3] >= 0 && !over) {
      const long long move = evalProgram(code + rec[3], values) -
          (rec[1] == 4 ? 0 : evalProgram(code + rec[3], capacities));
      *reinterpret_cast<unsigned char**>(slot) += move * slot[9];
    }
    if (rec[1] == 3) {
      continue;
    }
    const int rank = rec[2];
    int* dims = reinterpret_cast<int*>(slot + 12);
    int* strides = reinterpret_cast<int*>(slot + 28);
    // Whether the host laid the tensor out densely over its capacity dims;
    // a stride coded kDenseStride is then the dense stride of the real dims.
    bool hostDense = true;
    long long hostExpected = 1;
    for (int d = rank - 1; d >= 0; --d) {
      if (dims[d] != 1 && strides[d] != hostExpected) {
        hostDense = false;
      }
      hostExpected *= dims[d];
    }
    long long numEl = 1;
    for (int d = 0; d < rank; ++d) {
      // A dim coded kHostValue keeps what the host wrote.
      const long long dim = over ? 0
          : rec[4 + d] >= 0 ? evalProgram(code + rec[4 + d], values)
                            : dims[d];
      dims[d] = static_cast<int>(dim);
      if (rec[8 + d] >= 0) {
        strides[d] = static_cast<int>(evalProgram(code + rec[8 + d], values));
      }
      numEl *= dim;
    }
    long long dense = 1;
    for (int d = rank - 1; d >= 0; --d) {
      if (rec[8 + d] == kDenseStride) {
        strides[d] = hostDense ? static_cast<int>(dense) : 0;
      }
      dense *= dims[d] > 1 ? dims[d] : 1;
      // Broadcasting reads a size-1 dim through stride 0, as the host writes it.
      if (dims[d] == 1) {
        strides[d] = 0;
      }
    }
    bool contiguous = true;
    long long expected = 1;
    for (int d = rank - 1; d >= 0; --d) {
      if (dims[d] != 1 && strides[d] != expected) {
        contiguous = false;
      }
      expected *= dims[d];
    }
    // At most the capacity's element count, which the host wrote in the same
    // 32 bits.
    *reinterpret_cast<unsigned*>(slot + 44) = static_cast<unsigned>(numEl);
    *reinterpret_cast<unsigned*>(slot + 48) = 0;
    if (rec[1] != 2) {
      *reinterpret_cast<bool*>(slot + 52) = contiguous;
    }
  }
}

extern "C" __global__ void fixupSymbols(
    unsigned char* params,
    const int* records,
    const long long* code,
    int numRecords,
    const long long* symbols,
    int numSymbols,
    int numCode) {
  extern __shared__ long long fixupCode[];
  fixupBody(
      params,
      records,
      code,
      numRecords,
      symbols,
      numSymbols,
      numCode <= kSharedCode ? fixupCode : nullptr,
      numCode,
      0,
      blockDim.x,
      1);
}
)";

facebook::velox::wave::CompiledKernel& fixupKernel() {
  static std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel;
  static folly::once_flag flag;
  folly::call_once(flag, [] {
    kernel =
        facebook::velox::wave::CompiledKernel::getKernel("fixupSymbols", [] {
          facebook::velox::wave::KernelSpec spec;
          spec.code = kFixupSource;
          spec.entryPoints = {"fixupSymbols"};
          return spec;
        });
  });
  return *kernel;
}

// The symbols an expression list refers to.
void symbolsOf(const SymShape& shape, std::vector<std::string>& out) {
  for (const auto* list : {&shape.dims, &shape.strides}) {
    for (const auto& expr : *list) {
      expr.collectSymbols(out);
    }
  }
}

// Marks a symbol source that is dims[0] of a returned Tensor rather than a
// returned int64.
constexpr int64_t kDimsSource = int64_t{1} << 62;

// True when the shape of 'user''s output does not depend on 'input': a
// new_ones takes only its dtype from 'self', a gather reads 'self' at the
// positions its index gives, and an elementwise op takes a scalar operand as a
// value, which the fixups give the kernel. 'inputShape' is the input's shape.
bool shapeIgnores(
    const nativert::Node& user,
    const nativert::Value& input,
    const SymShape* inputShape) {
  static const folly::F14FastSet<std::string_view> kGathers{
      "tw.index_select",
      "tw.index_elt_one",
      "tw.index_elt_one_default",
      "torch.ops.aten.index_select.default",
  };
  if (kGathers.count(user.target()) != 0) {
    const auto* self = user.tryGetInput("self");
    if (self == nullptr || self->value != &input || inputShape == nullptr) {
      return false;
    }
    // Only the gathered dim may depend on the symbol.
    for (size_t d = 1; d < inputShape->dims.size(); ++d) {
      if (!inputShape->dims[d].isConstant()) {
        return false;
      }
    }
    if (user.target() == "tw.index_select" ||
        user.target() == "torch.ops.aten.index_select.default") {
      const auto* dim = user.tryGetAttribute("dim");
      return dim != nullptr && std::holds_alternative<int64_t>(dim->value) &&
          std::get<int64_t>(dim->value) == 0;
    }
    return true;
  }
  static const folly::F14FastSet<std::string_view> kNewFactories{
      "torch.ops.aten.new_ones.default",
      "torch.ops.aten.new_zeros.default",
      "torch.ops.aten.new_full.default",
      "torch.ops.aten.new_empty.default",
  };
  if (kNewFactories.count(user.target()) != 0) {
    const auto* self = user.tryGetInput("self");
    return self != nullptr && self->value == &input;
  }
  return input.type().kind() != nativert::Type::Kind::Tensor &&
      (keepsSelfShape(user.target()) || broadcastsOperands(user.target()));
}

// True for an integer read only by runtime checks (u1 >= 0), such as the
// sym_storage_offset of a slice.
bool onlyChecked(const nativert::Value& value) {
  if (value.type().kind() != nativert::Type::Kind::SymInt ||
      value.users().empty()) {
    return false;
  }
  for (const auto* user : value.users()) {
    for (const auto* output : user->outputs()) {
      if (output->type().kind() != nativert::Type::Kind::SymBool) {
        return false;
      }
    }
  }
  return true;
}

// True when 'expr' never decreases as 'symbol' grows, taking every symbol as
// non-negative. Sizing from a capacity is safe only then: a slice length
// Min(end, n) - Min(start, n) shrinks as its start grows.
bool nonDecreasingIn(const SymExpr& expr, const std::string& symbol, int sign) {
  switch (expr.kind) {
    case SymExpr::Kind::kConst:
      return true;
    case SymExpr::Kind::kSymbol:
      return expr.symbol != symbol || sign > 0;
    case SymExpr::Kind::kAdd:
    case SymExpr::Kind::kMax:
    case SymExpr::Kind::kMin:
      return std::all_of(
          expr.args.begin(), expr.args.end(), [&](const auto& a) {
            return nonDecreasingIn(a, symbol, sign);
          });
    case SymExpr::Kind::kMul: {
      int factorSign = sign;
      for (const auto& arg : expr.args) {
        if (arg.isConstant() && arg.value < 0) {
          factorSign = -factorSign;
        }
      }
      return std::all_of(
          expr.args.begin(), expr.args.end(), [&](const auto& a) {
            return nonDecreasingIn(a, symbol, factorSign);
          });
    }
    case SymExpr::Kind::kFloorDiv:
      return expr.args.size() == 2 &&
          nonDecreasingIn(expr.args[0], symbol, sign) &&
          nonDecreasingIn(expr.args[1], symbol, -sign);
    case SymExpr::Kind::kPow:
      return expr.args.size() == 2 && expr.args[1].isConstant() &&
          expr.args[1].value >= 0 &&
          nonDecreasingIn(expr.args[0], symbol, sign);
    case SymExpr::Kind::kMod: {
      std::vector<std::string> symbols;
      expr.collectSymbols(symbols);
      return std::find(symbols.begin(), symbols.end(), symbol) == symbols.end();
    }
  }
  return false;
}

bool isDataSymbol(const std::string& name) {
  return !name.empty() && name[0] == 'u';
}

// Replaces each symbol 'definitions' has by its expression.
void substituteSymbols(
    SymExpr& expr,
    const folly::F14FastMap<std::string, SymExpr>& definitions) {
  if (expr.kind == SymExpr::Kind::kSymbol) {
    auto it = definitions.find(expr.symbol);
    if (it != definitions.end()) {
      expr = it->second;
    }
    return;
  }
  for (auto& arg : expr.args) {
    substituteSymbols(arg, definitions);
  }
}

SymExpr binary(SymExpr::Kind kind, SymExpr lhs, SymExpr rhs) {
  SymExpr result{.kind = kind};
  result.args.push_back(std::move(lhs));
  result.args.push_back(std::move(rhs));
  return result;
}

void renameSymbols(
    SymExpr& expr,
    const folly::F14FastMap<std::string, std::string>& renames) {
  if (expr.kind == SymExpr::Kind::kSymbol) {
    auto it = renames.find(expr.symbol);
    if (it != renames.end()) {
      expr.symbol = it->second;
    }
  }
  for (auto& arg : expr.args) {
    renameSymbols(arg, renames);
  }
}

} // namespace

DeviceSymbols::DeviceSymbols(
    std::shared_ptr<SymShapes> shapes,
    WaveGraph& waveGraph)
    : waveGraph_(waveGraph) {
  inferSymShapes(*shapes, waveGraph_);
  shapes_ = std::make_shared<SymShapes>(*shapes);
  mergeEqualSymbols();
  for (const auto& [id, value] : waveGraph_.idToValue()) {
    auto it = shapes_->scalarByName.find(std::string(value->name()));
    if (it == shapes_->scalarByName.end() ||
        it->second.kind != SymExpr::Kind::kSymbol ||
        !isDataSymbol(it->second.symbol)) {
      continue;
    }
    candidateSymbol_.emplace(id, it->second.symbol);
  }
  // The count of a masked_select: the host reserves its output for every
  // element of the input, and the kernel writes the real count into the
  // output's dims in its parameters. An integer that carries the same symbol
  // (its sym_size) is computed on the host from those dims.
  folly::F14FastSet<std::string> carried;
  for (const auto& node : waveGraph_.graph()->nodes()) {
    if (node.target() != "torch.ops.aten.masked_select.default" ||
        node.outputs().size() != 1 || node.inputs().empty()) {
      continue;
    }
    const auto* output = node.outputs()[0];
    const auto* shape = shapeOf(output->id());
    if (shape == nullptr || shape->dims.size() != 1 ||
        shape->dims[0].kind != SymExpr::Kind::kSymbol ||
        !isDataSymbol(shape->dims[0].symbol) ||
        carried.count(shape->dims[0].symbol) != 0) {
      continue;
    }
    const auto* self = node.inputs()[0].value;
    auto dims = staticDimsOf(waveGraph_.types(), self);
    if (!dims || !isFullyStatic(waveGraph_.types(), self)) {
      continue;
    }
    int64_t bound = 1;
    for (auto dim : *dims) {
      bound *= dim;
    }
    const auto& symbol = shape->dims[0].symbol;
    carried.insert(symbol);
    candidateTensor_.emplace(output->id(), symbol);
    boundOf_[symbol] = bound;
    capacity_[symbol] = bound;
  }
}

void DeviceSymbols::mergeEqualSymbols() {
  // A broadcast between two non-1 dims proves them equal; the export names
  // the output's dim after one operand. Merging the two lets a fixup give the
  // output the value the device holds for the other.
  folly::F14FastSet<std::string> carried;
  for (const auto& [name, expr] : shapes_->scalarByName) {
    if (expr.kind == SymExpr::Kind::kSymbol && isDataSymbol(expr.symbol)) {
      carried.insert(expr.symbol);
    }
  }
  folly::F14FastMap<std::string, std::string> parent;
  std::function<std::string(const std::string&)> find =
      [&](const std::string& symbol) -> std::string {
    auto it = parent.find(symbol);
    if (it == parent.end() || it->second == symbol) {
      return symbol;
    }
    auto root = find(it->second);
    parent[symbol] = root;
    return root;
  };
  auto unite = [&](const std::string& lhs, const std::string& rhs) {
    auto rootA = find(lhs);
    auto rootB = find(rhs);
    if (rootA == rootB) {
      return;
    }
    // The root is a symbol some integer value carries, when one is.
    if (carried.count(rootA) == 0 && carried.count(rootB) != 0) {
      std::swap(rootA, rootB);
    }
    parent[rootB] = rootA;
  };
  auto shapeByValue = [&](const nativert::Value* value) -> const SymShape* {
    auto it = shapes_->byName.find(std::string(value->name()));
    return it != shapes_->byName.end() ? &it->second : nullptr;
  };
  for (const auto& node : waveGraph_.graph()->nodes()) {
    if ((!broadcastsOperands(node.target()) &&
         node.target() != "torch.ops.aten.expand.default") ||
        node.outputs().size() != 1) {
      continue;
    }
    const auto* out = shapeByValue(node.outputs()[0]);
    if (out == nullptr) {
      continue;
    }
    for (const auto& input : node.inputs()) {
      if (input.value == nullptr ||
          input.value->type().kind() != nativert::Type::Kind::Tensor) {
        continue;
      }
      const auto* in = shapeByValue(input.value);
      if (in == nullptr) {
        continue;
      }
      for (size_t i = 0; i < in->dims.size() && i < out->dims.size(); ++i) {
        const auto& a = out->dims[out->dims.size() - 1 - i];
        const auto& b = in->dims[in->dims.size() - 1 - i];
        if (a.kind == SymExpr::Kind::kSymbol &&
            b.kind == SymExpr::Kind::kSymbol && isDataSymbol(a.symbol) &&
            isDataSymbol(b.symbol) && a.symbol != b.symbol) {
          unite(a.symbol, b.symbol);
        }
      }
    }
  }
  folly::F14FastMap<std::string, std::string> renames;
  for (const auto& [symbol, unused] : parent) {
    auto root = find(symbol);
    if (root != symbol) {
      renames[symbol] = root;
    }
  }
  if (!renames.empty()) {
    applyRenames(renames);
  }
  defineSliceSymbols();
}

void DeviceSymbols::applyRenames(
    const folly::F14FastMap<std::string, std::string>& renames) {
  for (auto& [name, shape] : shapes_->byName) {
    for (auto* list : {&shape.dims, &shape.strides}) {
      for (auto& expr : *list) {
        renameSymbols(expr, renames);
      }
    }
  }
  for (auto& [name, expr] : shapes_->scalarByName) {
    renameSymbols(expr, renames);
  }
  if (std::getenv("TW_DEVICE_SYMBOLS_TRACE") != nullptr) {
    std::string text;
    for (const auto& [from, to] : renames) {
      text += fmt::format(" {}={}", from, to);
    }
    LOG(INFO) << "Device symbols merged:" << text;
  }
}

void DeviceSymbols::defineSliceSymbols() {
  // The export gives a slice with data-dependent bounds a fresh symbol for
  // its length. Defining it from the bounds lets the slice's output follow
  // them: Max(0, Min(end, n) - Min(start, n)) for step 1. A negative bound
  // would mean counting from the end; the bounds are symbols the fixup
  // kernel reads, and a negative one is reported like an exceeded capacity.
  auto boundOf = [&](const nativert::Node& node,
                     const char* name,
                     const SymExpr& absent) -> std::optional<SymExpr> {
    if (const auto* input = node.tryGetInput(name);
        input != nullptr && input->value != nullptr) {
      auto it = shapes_->scalarByName.find(std::string(input->value->name()));
      if (it == shapes_->scalarByName.end()) {
        return std::nullopt;
      }
      return it->second;
    }
    const auto* attr = node.tryGetAttribute(name);
    if (attr == nullptr ||
        std::holds_alternative<nativert::None>(attr->value)) {
      return absent;
    }
    if (std::holds_alternative<int64_t>(attr->value)) {
      return SymExpr::constant(std::get<int64_t>(attr->value));
    }
    return std::nullopt;
  };
  // A symbol an item() returns is the data's, not the slice's.
  folly::F14FastSet<std::string> itemSymbols;
  for (const auto& node : waveGraph_.graph()->nodes()) {
    if (node.target() != "torch.ops.aten._local_scalar_dense.default" ||
        node.outputs().size() != 1) {
      continue;
    }
    auto it =
        shapes_->scalarByName.find(std::string(node.outputs()[0]->name()));
    if (it != shapes_->scalarByName.end() &&
        it->second.kind == SymExpr::Kind::kSymbol) {
      itemSymbols.insert(it->second.symbol);
    }
  }
  folly::F14FastMap<std::string, SymExpr> definitions;
  for (const auto& node : waveGraph_.graph()->nodes()) {
    if (node.target() != "torch.ops.aten.slice.Tensor" ||
        node.outputs().size() != 1) {
      continue;
    }
    const auto* self = node.tryGetInput("self");
    if (self == nullptr || self->value == nullptr) {
      continue;
    }
    auto inIt = shapes_->byName.find(std::string(self->value->name()));
    auto outIt = shapes_->byName.find(std::string(node.outputs()[0]->name()));
    if (inIt == shapes_->byName.end() || outIt == shapes_->byName.end()) {
      continue;
    }
    const auto& in = inIt->second;
    const auto& out = outIt->second;
    int64_t dim = 0;
    if (const auto* attr = node.tryGetAttribute("dim");
        attr != nullptr && std::holds_alternative<int64_t>(attr->value)) {
      dim = std::get<int64_t>(attr->value);
    }
    if (dim < 0) {
      dim += static_cast<int64_t>(in.dims.size());
    }
    if (dim < 0 || dim >= static_cast<int64_t>(in.dims.size()) ||
        in.dims.size() != out.dims.size()) {
      continue;
    }
    const auto& length = out.dims[dim];
    if (length.kind != SymExpr::Kind::kSymbol || !isDataSymbol(length.symbol) ||
        itemSymbols.count(length.symbol) != 0 ||
        definitions.count(length.symbol) != 0) {
      continue;
    }
    auto step = boundOf(node, "step", SymExpr::constant(1));
    auto start = boundOf(node, "start", SymExpr::constant(0));
    auto end = boundOf(node, "end", in.dims[dim]);
    if (!step || !step->isConstant() || step->value != 1 || !start || !end) {
      continue;
    }
    const auto& n = in.dims[dim];
    std::vector<std::string> boundSymbols;
    n.collectSymbols(boundSymbols);
    start->collectSymbols(boundSymbols);
    end->collectSymbols(boundSymbols);
    if (std::find(boundSymbols.begin(), boundSymbols.end(), length.symbol) !=
        boundSymbols.end()) {
      continue;
    }
    definitions.emplace(
        length.symbol,
        binary(
            SymExpr::Kind::kMax,
            SymExpr::constant(0),
            binary(
                SymExpr::Kind::kAdd,
                binary(SymExpr::Kind::kMin, *end, n),
                binary(
                    SymExpr::Kind::kMul,
                    SymExpr::constant(-1),
                    binary(SymExpr::Kind::kMin, *start, n)))));
  }
  if (definitions.empty()) {
    return;
  }
  for (auto& [name, shape] : shapes_->byName) {
    for (auto* list : {&shape.dims, &shape.strides}) {
      for (auto& expr : *list) {
        substituteSymbols(expr, definitions);
      }
    }
  }
  for (auto& [name, expr] : shapes_->scalarByName) {
    substituteSymbols(expr, definitions);
  }
  if (std::getenv("TW_DEVICE_SYMBOLS_TRACE") != nullptr) {
    std::string text;
    for (const auto& [symbol, expr] : definitions) {
      text += fmt::format(" {}={}", symbol, expr.toString());
    }
    LOG(INFO) << "Device symbols defined:" << text;
  }
}

bool DeviceSymbols::ready() const {
  std::lock_guard<std::mutex> lock(mutex_);
  // The bounds are known up front; the observed values come after the first
  // execution.
  return capacity_.size() > boundOf_.size();
}

int32_t DeviceSymbols::symbolIndex(const std::string& symbol) {
  auto [it, inserted] =
      symbolIndex_.try_emplace(symbol, static_cast<int32_t>(symbols_.size()));
  if (inserted) {
    symbols_.push_back(symbol);
  }
  return it->second;
}

const SymExpr* FOLLY_NULLABLE
DeviceSymbols::scalarOf(nativert::ValueId id) const {
  auto valueIt = waveGraph_.idToValue().find(id);
  if (valueIt == waveGraph_.idToValue().end()) {
    return nullptr;
  }
  auto it = shapes_->scalarByName.find(std::string(valueIt->second->name()));
  return it != shapes_->scalarByName.end() ? &it->second : nullptr;
}

const SymShape* FOLLY_NULLABLE
DeviceSymbols::shapeOf(nativert::ValueId id) const {
  auto valueIt = waveGraph_.idToValue().find(id);
  if (valueIt == waveGraph_.idToValue().end()) {
    return nullptr;
  }
  auto it = shapes_->byName.find(std::string(valueIt->second->name()));
  return it != shapes_->byName.end() ? &it->second : nullptr;
}

std::vector<DeviceSymbols::HostConcat> DeviceSymbols::hostConcats(
    const ExecutionState& state) const {
  std::vector<HostConcat> concats;
  // Concats a kernel lays out: the host places the bands of the operands the
  // kernel writes in place, up to the first whose extent the kernel sets.
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      for (const auto& data : sv.kernels) {
        for (size_t o = 0; o < data.actualOutputDescs.size(); ++o) {
          const auto& desc = data.actualOutputDescs[o];
          if (desc.concatLayout == nullptr || data.invocation == nullptr) {
            continue;
          }
          const auto& layout = *desc.concatLayout;
          const auto& bindings = data.invocation->bindings();
          const auto [spec, dtype] = layout.resolve(data.invocation->nodeMap());
          HostConcat concat{
              .resultId = data.actualOutputs[o],
              .dim = spec.dim,
              .stack = spec.isStack};
          bool hostPlaces = true;
          for (const auto& info : layout.inputs) {
            auto it = bindings.find(info.valueId);
            const auto id = it != bindings.end() ? it->second : info.valueId;
            concat.operands.push_back(id);
            concat.bands.emplace_back();
            if (info.isSubgraphInput || info.isView) {
              continue;
            }
            hostPlaces = hostPlaces && !info.hasShapeOnDevice;
            if (hostPlaces) {
              concat.bands.back().push_back(id);
            }
          }
          concats.push_back(std::move(concat));
        }
      }
    }
  }
  // Concats an allocation group lays out: carved operands and the copies of
  // the others write bands the host places.
  for (const auto& node : waveGraph_.nodes()) {
    const auto* invocation = node->kernels();
    const auto* plan =
        invocation != nullptr ? invocation->allocGroupPlan() : nullptr;
    if (plan == nullptr) {
      continue;
    }
    for (const auto& group : plan->groups) {
      if (group.concat == nullptr) {
        continue;
      }
      const auto& layout = *group.concat;
      HostConcat concat{
          .resultId = layout.resultId,
          .dim = layout.dim,
          .stack = layout.isStack};
      for (size_t i = 0; i < layout.operands.size(); ++i) {
        const auto& operand = layout.operands[i];
        concat.operands.push_back(operand.valueId);
        concat.bands.emplace_back();
        if (layout.memberOfOperand[i] >= 0) {
          concat.bands.back().push_back(operand.valueId);
        }
        if (operand.copyDestId >= 0) {
          concat.bands.back().push_back(operand.copyDestId);
        }
      }
      concats.push_back(std::move(concat));
    }
  }
  return concats;
}

std::string DeviceSymbols::bandMovable(
    nativert::ValueId band,
    const folly::F14FastSet<nativert::ValueId>& hostRead,
    const folly::F14FastSet<nativert::ValueId>& graphOutputs) const {
  if (hostRead.count(band) != 0 || graphOutputs.count(band) != 0) {
    return fmt::format("band %{} is read on host or handed out", band);
  }
  // A view of a band takes its address from the host's band.
  auto valueIt = waveGraph_.idToValue().find(band);
  if (valueIt == waveGraph_.idToValue().end()) {
    return "";
  }
  for (const auto* user : valueIt->second->users()) {
    const auto* meta = Registry::metadata(user->target());
    if (user->target() != "prim.ListPack" &&
        (meta == nullptr || meta->isView())) {
      return fmt::format("band %{} has a view {}", band, user->target());
    }
  }
  return "";
}

void DeviceSymbols::traceRejection(const std::string& why) const {
  if (why.rfind('%', 0) != 0) {
    return;
  }
  const auto id = static_cast<nativert::ValueId>(std::stoi(why.substr(1)));
  auto valueIt = waveGraph_.idToValue().find(id);
  if (valueIt == waveGraph_.idToValue().end() ||
      valueIt->second->producer() == nullptr) {
    return;
  }
  auto describe = [&](nativert::ValueId value) {
    std::string text;
    if (const auto* shape = shapeOf(value)) {
      for (size_t d = 0; d < shape->dims.size(); ++d) {
        text += fmt::format(
            " {}/{}", shape->dims[d].toString(), shape->strides[d].toString());
      }
    } else if (const auto* scalar = scalarOf(value)) {
      text = " = " + scalar->toString();
    }
    return text;
  };
  const auto* producer = valueIt->second->producer();
  LOG(INFO) << "  %" << id << " = " << producer->target() << describe(id);
  for (const auto& input : producer->inputs()) {
    if (input.value == nullptr) {
      continue;
    }
    LOG(INFO) << "    " << input.name << " %" << input.value->id()
              << describe(input.value->id());
    const auto* pack = input.value->producer();
    if (pack == nullptr || pack->target() != "prim.ListPack") {
      continue;
    }
    for (const auto& element : pack->inputs()) {
      if (element.value != nullptr) {
        const auto* elementProducer = element.value->producer();
        LOG(INFO) << "      element %" << element.value->id() << " "
                  << (elementProducer ? elementProducer->target() : "-")
                  << describe(element.value->id());
      }
    }
  }
  for (const auto* user : valueIt->second->users()) {
    LOG(INFO) << "    user " << user->target() << " -> %"
              << (user->outputs().empty() ? -1 : user->outputs()[0]->id());
  }
}

void DeviceSymbols::selectSubstitutions(
    ExecutionState& state,
    folly::F14FastSet<int32_t>& returned) {
  // Values the host reads: an eager or host op's operands, host copies,
  // returned tensors, graph outputs.
  folly::F14FastSet<nativert::ValueId> hostRead;
  folly::F14FastSet<nativert::ValueId> returnedLists;
  // Outputs whose dims a kernel sets; each is returned itself.
  folly::F14FastSet<nativert::ValueId> deviceShaped;
  // Where each returned value comes back from: (node, step, kernel, return).
  folly::F14FastMap<nativert::ValueId, std::tuple<int32_t, int32_t, int64_t>>
      sourceOf;
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    const auto& steps = state.stepVectors[node];
    for (size_t step = 0; step < steps.size(); ++step) {
      const auto& sv = steps[step];
      // A metadata-only op on the host sizes from the capacity like the
      // rest of the host does; an eager op computes on what it reads.
      for (const auto& data : sv.standalones) {
        if (data.launch == nullptr || !data.launch->metadataOnly) {
          hostRead.insert(data.actualInputs.begin(), data.actualInputs.end());
        }
      }
      for (const auto& copy : sv.hostCopies) {
        hostRead.insert(copy.id);
      }
      for (size_t k = 0; k < sv.kernels.size(); ++k) {
        const auto& data = sv.kernels[k];
        for (size_t i = 0; i < data.actualOutputDescs.size(); ++i) {
          if (data.actualOutputDescs[i].shapeSetOnDevice) {
            deviceShaped.insert(data.actualOutputs[i]);
          }
        }
        for (size_t r = 0; r < data.returnValues.size(); ++r) {
          if (data.returnTypes[r] == nativert::Type::Kind::TensorList) {
            // Brings back its elements' dims; see the closure below.
            returnedLists.insert(data.returnValues[r]);
            continue;
          }
          const auto where =
              (static_cast<int64_t>(k) << 32) | static_cast<int64_t>(r);
          // The bound is the capacity only where the kernel sets the count
          // in a buffer the host reserved for every input element; a
          // multi-kernel masked_select sizes its final from the total.
          if (data.returnTypes[r] == nativert::Type::Kind::Tensor &&
              candidateTensor_.count(data.returnValues[r]) != 0 &&
              deviceShaped.count(data.returnValues[r]) != 0) {
            sourceOf[data.returnValues[r]] = {
                static_cast<int32_t>(node),
                static_cast<int32_t>(step),
                where | kDimsSource};
            continue;
          }
          if (data.returnTypes[r] != nativert::Type::Kind::SymInt) {
            hostRead.insert(data.returnValues[r]);
            continue;
          }
          sourceOf[data.returnValues[r]] = {
              static_cast<int32_t>(node), static_cast<int32_t>(step), where};
        }
      }
    }
  }
  // A graph output in a region gets its real dims when the execution ends.
  folly::F14FastSet<nativert::ValueId> graphOutputs;
  for (const auto* output : waveGraph_.graph()->outputs()) {
    if (output != nullptr) {
      graphOutputs.insert(output->id());
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // Candidate symbol -> the value that carries it.
  folly::F14FastMap<std::string, nativert::ValueId> carrier;
  for (const auto* candidates : {&candidateSymbol_, &candidateTensor_}) {
    for (const auto& [id, symbol] : *candidates) {
      if (returned.count(id) != 0 && sourceOf.count(id) != 0 &&
          hostRead.count(id) == 0 && capacity_.count(symbol) != 0 &&
          banned_.count(symbol) == 0) {
        carrier.emplace(symbol, id);
      }
    }
  }
  // Other values carrying a candidate symbol (a repeat total equal to an
  // item()) run on the same capacity, and the region grows from them too.
  folly::F14FastMap<std::string, std::vector<nativert::ValueId>> extraCarriers;
  for (const auto& [id, symbol] : candidateSymbol_) {
    auto it = carrier.find(symbol);
    if (it != carrier.end() && it->second != id && returned.count(id) != 0 &&
        hostRead.count(id) == 0) {
      extraCarriers[symbol].push_back(id);
    }
  }
  // The values shaped by each candidate symbol.
  folly::F14FastMap<std::string, std::vector<nativert::ValueId>> region;
  folly::F14FastMap<std::string, std::vector<nativert::ValueId>> regionOutputs;
  // Returned lists in a region whose return the pick makes unneeded.
  folly::F14FastMap<std::string, std::vector<nativert::ValueId>> regionLists;
  folly::F14FastMap<std::string, folly::F14FastSet<std::string>> needs;
  // Concat bands in a region whose start moves with the symbol, and the start.
  folly::F14FastMap<
      std::string,
      std::vector<std::pair<nativert::ValueId, SymExpr>>>
      regionBands;
  folly::F14FastSet<std::string> rejected;
  folly::F14FastMap<std::string, std::string> reason;
  auto reject = [&](const std::string& symbol, std::string why) {
    if (rejected.insert(symbol).second) {
      reason[symbol] = std::move(why);
    }
  };
  for (const auto& [id, symbol] : candidateTensor_) {
    if (carrier.count(symbol) == 0) {
      reason[symbol] = returned.count(id) == 0 ? "count not returned"
          : sourceOf.count(id) == 0            ? "count has no kernel source"
                                               : "count read on host";
    }
  }
  for (const auto& [id, symbol] : candidateSymbol_) {
    if (carrier.count(symbol) == 0 && reason.count(symbol) == 0) {
      reason[symbol] = returned.count(id) == 0 ? "not returned"
          : sourceOf.count(id) == 0            ? "no kernel source"
          : hostRead.count(id) != 0            ? "read on host"
                                               : "not observed yet";
    }
  }
  // Slices the host runs as shortcuts, which can view the whole extent.
  folly::F14FastMap<nativert::ValueId, LaunchData*> sliceShortcuts;
  for (auto& steps : state.stepVectors) {
    for (auto& sv : steps) {
      for (auto& data : sv.shortcutStandalones) {
        if (data.launch != nullptr &&
            data.launch->standaloneShortcut == StandaloneShortcut::kSlice &&
            data.actualOutputs.size() == 1) {
          data.wholeExtent = false;
          sliceShortcuts.emplace(data.actualOutputs[0], &data);
        }
      }
    }
  }
  // A slice whose start the device decides views the whole extent of its
  // base on the host, a bound for its length; the fixups move it to its real
  // start. Its offset in elements, when 'id' is a slice with a data-dependent
  // start or a length the capacities may shrink, whose values only copies
  // read.
  auto deviceViewStart = [&](nativert::ValueId id) -> std::optional<SymExpr> {
    const auto* value = waveGraph_.idToValue().at(id);
    const auto* producer = value->producer();
    if (producer == nullptr ||
        producer->target() != "torch.ops.aten.slice.Tensor" ||
        sliceShortcuts.count(id) == 0) {
      return std::nullopt;
    }
    auto intOf = [&](const char* name,
                     int64_t absent) -> std::optional<SymExpr> {
      if (const auto* attr = producer->tryGetAttribute(name)) {
        if (std::holds_alternative<int64_t>(attr->value)) {
          return SymExpr::constant(std::get<int64_t>(attr->value));
        }
        return std::holds_alternative<nativert::None>(attr->value)
            ? std::optional<SymExpr>(SymExpr::constant(absent))
            : std::nullopt;
      }
      const auto* input = producer->tryGetInput(name);
      if (input == nullptr || input->value == nullptr) {
        return SymExpr::constant(absent);
      }
      const auto* scalar = scalarOf(input->value->id());
      return scalar != nullptr ? std::optional<SymExpr>(*scalar) : std::nullopt;
    };
    const auto* self = producer->tryGetInput("self");
    const auto* selfShape = self != nullptr && self->value != nullptr
        ? shapeOf(self->value->id())
        : nullptr;
    const auto dim = intOf("dim", 0);
    const auto step = intOf("step", 1);
    const auto start = intOf("start", 0);
    if (selfShape == nullptr || !dim.has_value() || !dim->isConstant() ||
        !step.has_value() || !step->isConstant() || step->value != 1 ||
        !start.has_value() || dim->value < 0 ||
        dim->value >= static_cast<int64_t>(selfShape->dims.size())) {
      return std::nullopt;
    }
    std::vector<std::string> startSymbols;
    start->collectSymbols(startSymbols);
    bool needed =
        std::any_of(startSymbols.begin(), startSymbols.end(), isDataSymbol);
    if (const auto* shape = shapeOf(id); !needed && shape != nullptr) {
      for (const auto& dimExpr : shape->dims) {
        std::vector<std::string> names;
        dimExpr.collectSymbols(names);
        for (const auto& name : names) {
          needed = needed ||
              (isDataSymbol(name) && !nonDecreasingIn(dimExpr, name, 1));
        }
      }
    }
    if (!needed) {
      return std::nullopt;
    }
    static const folly::F14FastSet<std::string_view> kCopies{
        "torch.ops.aten._to_copy.default",
        "torch.ops.aten.clone.default",
    };
    for (const auto* user : value->users()) {
      if (kCopies.count(user->target()) != 0 ||
          user->target() == "torch.ops.aten._assert_tensor_metadata.default") {
        continue;
      }
      const bool checkedScalar = !user->outputs().empty() &&
          std::all_of(
              user->outputs().begin(),
              user->outputs().end(),
              [](const auto* output) { return onlyChecked(*output); });
      if (!checkedScalar) {
        return std::nullopt;
      }
    }
    return binary(
        SymExpr::Kind::kMul,
        binary(SymExpr::Kind::kMin, *start, selfShape->dims[dim->value]),
        selfShape->strides[dim->value]);
  };
  // Whether the host's dims of 'id' bound its real dims though its
  // expression is not monotone: a slice viewing the whole extent, a copy of
  // one, or an integer only runtime checks read.
  auto boundedOnHost = [&](nativert::ValueId id) {
    const auto* value = waveGraph_.idToValue().at(id);
    if (onlyChecked(*value) || deviceViewStart(id).has_value()) {
      return true;
    }
    const auto* producer = value->producer();
    return producer != nullptr && producer->inputs().size() >= 1 &&
        producer->inputs()[0].value != nullptr &&
        (producer->target() == "torch.ops.aten._to_copy.default" ||
         producer->target() == "torch.ops.aten.clone.default") &&
        deviceViewStart(producer->inputs()[0].value->id()).has_value();
  };
  // A symbol's region is what the host sizes from its capacity: the values
  // reachable from its carrier through ops whose outputs the symbol shapes.
  // Other values shaped by the symbol (or one merged with it) are sized from
  // the real value the host waits for elsewhere.
  for (const auto& [symbol, carrierId] : carrier) {
    auto& members = region[symbol];
    folly::F14FastSet<nativert::ValueId> seen{carrierId};
    members.push_back(carrierId);
    for (auto extra : extraCarriers[symbol]) {
      if (seen.insert(extra).second) {
        members.push_back(extra);
      }
    }
    for (size_t next = 0; next < members.size(); ++next) {
      const auto id = members[next];
      std::vector<std::string> symbols;
      if (const auto* shape = shapeOf(id)) {
        symbolsOf(*shape, symbols);
      } else if (const auto* scalar = scalarOf(id)) {
        scalar->collectSymbols(symbols);
      }
      needs[symbol].insert(symbols.begin(), symbols.end());
      const auto* memberShape = shapeOf(id);
      const auto* memberScalar = memberShape ? nullptr : scalarOf(id);
      bool monotone = true;
      if (memberShape != nullptr) {
        for (const auto& dim : memberShape->dims) {
          monotone = monotone && nonDecreasingIn(dim, symbol, 1);
        }
      } else if (memberScalar != nullptr) {
        monotone = nonDecreasingIn(*memberScalar, symbol, 1);
      }
      if (!monotone && !boundedOnHost(id)) {
        reject(symbol, fmt::format("%{} shrinks as the symbol grows", id));
      }
      if (hostRead.count(id) != 0) {
        reject(symbol, fmt::format("%{} read on host", id));
      }
      if (graphOutputs.count(id) != 0) {
        if (shapeOf(id) == nullptr) {
          reject(symbol, fmt::format("%{} is a graph output", id));
        }
        regionOutputs[symbol].push_back(id);
      }
      const auto* value = waveGraph_.idToValue().at(id);
      if (returnedLists.count(id) != 0) {
        // The return resizes the elements from their device dims. An element
        // in the region has its capacity dims on the host and its real ones
        // from a fixup; one the host sizes itself has them already. Another
        // element's dims come only from the return.
        const auto* pack = value->producer();
        bool hostSized = pack != nullptr && pack->target() == "prim.ListPack";
        for (size_t i = 0; hostSized && i < pack->inputs().size(); ++i) {
          const auto* element = pack->inputs()[i].value;
          if (element == nullptr || seen.count(element->id()) != 0) {
            continue;
          }
          const auto* elementShape = shapeOf(element->id());
          std::vector<std::string> elementSymbols;
          if (elementShape != nullptr) {
            symbolsOf(*elementShape, elementSymbols);
          }
          hostSized =
              returned.count(element->id()) == 0 &&
              std::none_of(
                  elementSymbols.begin(), elementSymbols.end(), isDataSymbol) &&
              (elementShape != nullptr ||
               isFullyStatic(waveGraph_.types(), element));
        }
        if (!hostSized) {
          reject(symbol, fmt::format("list %{} returns device dims", id));
        }
        regionLists[symbol].push_back(id);
      }
      for (const auto* user : value->users()) {
        for (const auto* output : user->outputs()) {
          // A users() list can keep a node a merge has since removed.
          if (waveGraph_.idToValue().count(output->id()) == 0) {
            continue;
          }
          const auto kind = output->type().kind();
          std::vector<std::string> outSymbols;
          if (const auto* outShape = shapeOf(output->id())) {
            symbolsOf(*outShape, outSymbols);
          } else if (const auto* outScalar = scalarOf(output->id())) {
            outScalar->collectSymbols(outSymbols);
          } else if (kind == nativert::Type::Kind::Tensor) {
            // Scratch nothing reads, such as a scan's per-block partials,
            // is sized by the op itself.
            if (!isFullyStatic(waveGraph_.types(), output) &&
                (!output->users().empty() ||
                 graphOutputs.count(output->id()) != 0)) {
              reject(
                  symbol,
                  fmt::format("no shape for the output of {}", user->target()));
            }
            continue;
          } else if (
              kind == nativert::Type::Kind::SymInt ||
              kind == nativert::Type::Kind::SymFloat) {
            reject(
                symbol,
                fmt::format(
                    "no expression for the scalar output of {}",
                    user->target()));
            continue;
          } else if (
              kind == nativert::Type::Kind::None ||
              kind == nativert::Type::Kind::SymBool) {
            // A SymBool is a runtime check on the symbol (u1 >= 0), which
            // holds for the capacity as well.
            continue;
          } else {
            // A list carries its elements' shapes to its users.
            if (seen.insert(output->id()).second) {
              members.push_back(output->id());
            }
            continue;
          }
          if (std::find(outSymbols.begin(), outSymbols.end(), symbol) !=
              outSymbols.end()) {
            if (seen.insert(output->id()).second) {
              members.push_back(output->id());
            }
          } else if (
              !outSymbols.empty() &&
              !shapeIgnores(*user, *value, shapeOf(id)) &&
              !onlyChecked(*output)) {
            // E.g. a broadcast with an operand of an unrelated symbol: sized
            // on the host from the capacity, with no fixup.
            reject(
                symbol,
                fmt::format(
                    "%{} {} gives %{} other symbols",
                    id,
                    user->target(),
                    output->id()));
          }
        }
      }
    }
  }
  // A region member's producer must not mix capacity and real sizes: each of
  // its tensor operands that the symbol shapes has to be in the region as
  // well. An expand of a really-sized tensor to a capacity size fails on the
  // host; a broadcast of the two reads past the smaller one.
  for (const auto& [symbol, members] : region) {
    if (rejected.count(symbol) != 0) {
      continue;
    }
    folly::F14FastSet<nativert::ValueId> inRegion(
        members.begin(), members.end());
    for (auto id : members) {
      const auto* producer = waveGraph_.idToValue().at(id)->producer();
      if (producer == nullptr || carrier.at(symbol) == id) {
        continue;
      }
      for (const auto& input : producer->inputs()) {
        if (input.value == nullptr ||
            input.value->type().kind() != nativert::Type::Kind::Tensor ||
            inRegion.count(input.value->id()) != 0 ||
            shapeIgnores(*producer, *input.value, shapeOf(input.value->id()))) {
          continue;
        }
        const auto* inputShape = shapeOf(input.value->id());
        if (inputShape == nullptr) {
          continue;
        }
        std::vector<std::string> inputSymbols;
        symbolsOf(*inputShape, inputSymbols);
        if (std::find(inputSymbols.begin(), inputSymbols.end(), symbol) !=
            inputSymbols.end()) {
          reject(
              symbol,
              fmt::format(
                  "%{} {} mixes really sized %{}",
                  id,
                  producer->target(),
                  input.value->id()));
          break;
        }
      }
      if (rejected.count(symbol) != 0) {
        break;
      }
    }
  }
  // A view that moves the data pointer along a dim (select, slice, split)
  // takes its offset from the host's stride for that dim. When the stride
  // depends on the symbol, the host's capacity stride puts the view where the
  // kernels, writing with the real stride, did not.
  static const folly::F14FastSet<std::string_view> kOffsetViews{
      "torch.ops.aten.select.int",
      "torch.ops.aten.slice.Tensor",
      "torch.ops.aten.narrow.default",
      "torch.ops.aten.unbind.int",
      "torch.ops.aten.split.Tensor",
      "torch.ops.aten.split_with_sizes.default",
      "torch.ops.aten.as_strided.default",
  };
  for (const auto& [symbol, members] : region) {
    if (rejected.count(symbol) != 0) {
      continue;
    }
    for (auto id : members) {
      const auto* value = waveGraph_.idToValue().at(id);
      for (const auto* user : value->users()) {
        if (kOffsetViews.count(user->target()) == 0) {
          continue;
        }
        const auto* shape = shapeOf(id);
        if (shape == nullptr) {
          continue;
        }
        int64_t dim = 0;
        if (const auto* attr = user->tryGetAttribute("dim");
            attr != nullptr && std::holds_alternative<int64_t>(attr->value)) {
          dim = std::get<int64_t>(attr->value);
        }
        if (dim < 0) {
          dim += static_cast<int64_t>(shape->strides.size());
        }
        std::vector<std::string> strideSymbols;
        if (user->target() == "torch.ops.aten.as_strided.default") {
          symbolsOf(*shape, strideSymbols);
        } else if (
            dim >= 0 && dim < static_cast<int64_t>(shape->strides.size())) {
          shape->strides[dim].collectSymbols(strideSymbols);
        }
        if (std::find(strideSymbols.begin(), strideSymbols.end(), symbol) !=
            strideSymbols.end()) {
          reject(
              symbol,
              fmt::format(
                  "%{} {} offset uses a symbolic stride", id, user->target()));
          break;
        }
      }
      if (rejected.count(symbol) != 0) {
        break;
      }
    }
  }
  // A view's offset also moves with its start: a slice, narrow, select or
  // split whose start depends on the symbol gets a data pointer from the
  // capacity.
  auto inputSymbols = [&](const nativert::Node& node, const char* name) {
    std::vector<std::string> names;
    const auto* input = node.tryGetInput(name);
    if (input == nullptr || input->value == nullptr) {
      return names;
    }
    auto collect = [&](const nativert::Value* scalarValue) {
      if (const auto* scalar = scalarOf(scalarValue->id())) {
        scalar->collectSymbols(names);
      }
    };
    const auto* pack = input->value->producer();
    if (pack != nullptr && pack->target() == "prim.ListPack") {
      for (const auto& element : pack->inputs()) {
        if (element.value != nullptr) {
          collect(element.value);
        }
      }
    } else {
      collect(input->value);
    }
    return names;
  };
  for (const auto& [symbol, members] : region) {
    if (rejected.count(symbol) != 0) {
      continue;
    }
    for (auto id : members) {
      const auto* producer = waveGraph_.idToValue().at(id)->producer();
      if (producer == nullptr || kOffsetViews.count(producer->target()) == 0 ||
          deviceViewStart(id).has_value()) {
        continue;
      }
      std::vector<std::string> placement;
      for (const auto* name : {"start", "index", "split_size", "split_sizes"}) {
        auto names = inputSymbols(*producer, name);
        placement.insert(placement.end(), names.begin(), names.end());
      }
      if (std::find(placement.begin(), placement.end(), symbol) !=
          placement.end()) {
        reject(
            symbol,
            fmt::format(
                "%{} {} places data by the symbol", id, producer->target()));
        break;
      }
    }
  }
  // A concatenation places each operand after the ones before it. A band the
  // host places after an operand sized by a symbol gets its real start from
  // the fixups; the concat kernels place the rest from the real dims.
  for (const auto& concat : hostConcats(state)) {
    std::vector<std::string> extentSymbols;
    std::vector<std::pair<nativert::ValueId, SymExpr>> moved;
    std::string why;
    std::string unknownExtent;
    auto joinStride = SymExpr::constant(1);
    bool haveStride = false;
    for (const auto operand : concat.operands) {
      const auto* shape = shapeOf(operand);
      const auto rank =
          shape != nullptr ? static_cast<int64_t>(shape->dims.size()) : -1;
      if (shape != nullptr && rank > concat.dim - (concat.stack ? 1 : 0)) {
        for (auto d = concat.stack ? concat.dim : concat.dim + 1; d < rank;
             ++d) {
          joinStride = binary(SymExpr::Kind::kMul, joinStride, shape->dims[d]);
        }
        haveStride = true;
        break;
      }
    }
    auto start = SymExpr::constant(0);
    for (size_t i = 0; i < concat.operands.size(); ++i) {
      for (auto band : concat.bands[i]) {
        auto offset = binary(SymExpr::Kind::kMul, start, joinStride);
        std::vector<std::string> names;
        offset.collectSymbols(names);
        if (!unknownExtent.empty()) {
          why = unknownExtent;
        } else if (std::any_of(names.begin(), names.end(), isDataSymbol)) {
          if (!haveStride) {
            why = "has no operand shape";
          } else if (why.empty()) {
            why = bandMovable(band, hostRead, graphOutputs);
          }
          moved.emplace_back(band, std::move(offset));
        }
      }
      if (concat.stack) {
        start = SymExpr::constant(static_cast<int64_t>(i) + 1);
        continue;
      }
      const auto* shape = shapeOf(concat.operands[i]);
      if (shape == nullptr ||
          concat.dim >= static_cast<int64_t>(shape->dims.size())) {
        if (unknownExtent.empty()) {
          unknownExtent =
              fmt::format("operand %{} has no shape", concat.operands[i]);
        }
        continue;
      }
      shape->dims[concat.dim].collectSymbols(extentSymbols);
      start = binary(SymExpr::Kind::kAdd, start, shape->dims[concat.dim]);
    }
    if (!why.empty()) {
      for (const auto& symbol : extentSymbols) {
        reject(symbol, fmt::format("concat %{} {}", concat.resultId, why));
      }
      continue;
    }
    for (const auto& [band, offset] : moved) {
      std::vector<std::string> names;
      offset.collectSymbols(names);
      for (const auto& symbol : names) {
        if (carrier.count(symbol) != 0) {
          needs[symbol].insert(names.begin(), names.end());
          regionBands[symbol].emplace_back(band, offset);
        }
      }
    }
  }
  // A region's expressions may use only symbols that are themselves picked.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& [symbol, needed] : needs) {
      if (rejected.count(symbol) != 0) {
        continue;
      }
      for (const auto& other : needed) {
        if (carrier.count(other) == 0 || rejected.count(other) != 0) {
          reject(symbol, fmt::format("needs {}", other));
          changed = true;
          break;
        }
      }
    }
  }
  reason_ = reason;
  if (std::getenv("TW_DEVICE_SYMBOLS_TRACE") != nullptr) {
    for (const auto& [symbol, id] : carrier) {
      if (rejected.count(symbol) == 0) {
        LOG(INFO) << "Device symbols picked: " << symbol << " %" << id;
        for (auto member : region[symbol]) {
          const auto* shape = shapeOf(member);
          std::string dims;
          const auto* scalar = scalarOf(member);
          if (shape == nullptr && scalar != nullptr) {
            dims = " = " + scalar->toString();
          }
          for (size_t d = 0; shape != nullptr && d < shape->dims.size(); ++d) {
            dims += fmt::format(
                " {}/{}",
                shape->dims[d].toString(),
                shape->strides[d].toString());
          }
          const auto* value = waveGraph_.idToValue().at(member);
          LOG(INFO) << "  region %" << member << " "
                    << (value->producer() ? value->producer()->target() : "-")
                    << dims;
        }
      }
    }
    folly::F14FastMap<std::string, int32_t> counts;
    for (const auto& [symbol, why] : reason) {
      if (why.rfind('%', 0) == 0) {
        ++counts[why.substr(why.find(' ') + 1)];
      } else {
        ++counts[why];
      }
    }
    for (const auto& [why, count] : counts) {
      LOG(INFO) << "Device symbols rejected: " << count << " x " << why;
    }
    for (const auto& [symbol, why] : reason) {
      LOG(INFO) << "Device symbols reason: " << symbol << ": " << why;
      traceRejection(why);
    }
  }
  auto unreturn = [&](nativert::ValueId id) {
    if (returned.erase(id) != 0) {
      state.symbolUnreturned.insert(id);
    }
  };
  state.symbolUnreturned.clear();
  state.substitutedSymbol.clear();
  state.symbolRegion.clear();
  state.symbolOutputs.clear();
  state.capacityScalars.clear();
  state.numUnreturnedLists = 0;
  state.symbolSources.clear();
  state.bandStarts.clear();
  state.viewStarts.clear();
  // For bisecting: a comma-separated list of the symbols to pick, or empty.
  const char* only = std::getenv("TW_DEVICE_SYMBOLS_ONLY");
  const std::string onlyList = only != nullptr ? fmt::format(",{},", only) : "";
  for (const auto& [symbol, id] : carrier) {
    if (!onlyList.empty() &&
        onlyList.find(fmt::format(",{},", symbol)) == std::string::npos) {
      continue;
    }
    if (rejected.count(symbol) != 0 ||
        static_cast<int32_t>(symbols_.size()) >= kMaxSymbols) {
      continue;
    }
    const auto index = symbolIndex(symbol);
    state.substitutedSymbol[id] = index;
    state.symbolRegion.insert(region[symbol].begin(), region[symbol].end());
    for (auto list : regionLists[symbol]) {
      unreturn(list);
    }
    state.symbolOutputs.insert(
        state.symbolOutputs.end(),
        regionOutputs[symbol].begin(),
        regionOutputs[symbol].end());
    for (const auto& [band, start] : regionBands[symbol]) {
      state.bandStarts.emplace(band, start);
    }
    for (auto member : region[symbol]) {
      if (auto start = deviceViewStart(member)) {
        state.viewStarts.emplace(member, std::move(*start));
        sliceShortcuts.at(member)->wholeExtent = true;
      }
    }
    if (state.symbolSources.size() <= static_cast<size_t>(index)) {
      state.symbolSources.resize(index + 1, {-1, -1, -1});
    }
    // The fixups read the value from whichever carrier the device produces
    // first: a region member downstream of a later carrier can run before it.
    auto source = sourceOf.at(id);
    unreturn(id);
    for (auto extra : extraCarriers[symbol]) {
      state.substitutedSymbol[extra] = index;
      unreturn(extra);
      auto it = sourceOf.find(extra);
      if (it != sourceOf.end() &&
          std::make_pair(std::get<0>(it->second), std::get<1>(it->second)) <
              std::make_pair(std::get<0>(source), std::get<1>(source))) {
        source = it->second;
      }
    }
    state.symbolSources.at(index) = source;
  }
  // A returned integer in a region whose expression uses only picked symbols
  // (u1 + u2) takes its value from the capacities as well; the fixups give
  // the kernels its real value.
  folly::F14FastSet<std::string> pickedSymbols;
  for (const auto& [id, index] : state.substitutedSymbol) {
    pickedSymbols.insert(symbols_[index]);
  }
  for (const auto& [id, source] : sourceOf) {
    if (returned.count(id) == 0 || hostRead.count(id) != 0) {
      continue;
    }
    const auto* scalar = scalarOf(id);
    if (scalar == nullptr) {
      continue;
    }
    std::vector<std::string> symbols;
    scalar->collectSymbols(symbols);
    if (scalar->isConstant() && candidateTensor_.count(id) == 0 &&
        candidateSymbol_.count(id) == 0) {
      // The export fixed it: nothing has to wait for the device to say so.
      state.capacityScalars.insert(id);
      unreturn(id);
      continue;
    }
    if (state.symbolRegion.count(id) != 0 && !symbols.empty() &&
        std::all_of(symbols.begin(), symbols.end(), [&](const auto& name) {
          return pickedSymbols.count(name) != 0;
        })) {
      state.capacityScalars.insert(id);
      unreturn(id);
    }
  }
  // A packed list's return resizes its elements from their device dims. When
  // the host sized every element itself, the return repeats what the host
  // has, and the list's readers need not wait for it.
  for (auto list : returnedLists) {
    const auto* pack = waveGraph_.idToValue().at(list)->producer();
    if (returned.count(list) == 0 || pack == nullptr ||
        pack->target() != "prim.ListPack") {
      continue;
    }
    const bool hostSized = std::all_of(
        pack->inputs().begin(), pack->inputs().end(), [&](const auto& input) {
          return input.value != nullptr &&
              returned.count(input.value->id()) == 0 &&
              deviceShaped.count(input.value->id()) == 0;
        });
    if (hostSized) {
      unreturn(list);
      ++state.numUnreturnedLists;
    }
  }
}

int32_t DeviceSymbols::emitProgram(
    const SymExpr& expr,
    std::vector<int64_t>& code) {
  const auto begin = static_cast<int32_t>(code.size());
  int32_t depth = 0;
  auto push = [&](int64_t instruction) {
    code.push_back(instruction);
    ++depth;
    TORCH_CHECK(
        depth <= kProgramStackDepth,
        "Symbolic expression is deeper than the device evaluator's stack: ",
        expr.toString());
  };
  std::function<void(const SymExpr&)> emit = [&](const SymExpr& node) {
    switch (node.kind) {
      case SymExpr::Kind::kConst:
        push(word(kPushConst, node.value));
        return;
      case SymExpr::Kind::kSymbol:
        push(word(kPushSymbol, symbolIndex_.at(node.symbol)));
        return;
      case SymExpr::Kind::kAdd:
      case SymExpr::Kind::kMul:
      case SymExpr::Kind::kMax:
      case SymExpr::Kind::kMin:
      case SymExpr::Kind::kPow:
      case SymExpr::Kind::kFloorDiv:
      case SymExpr::Kind::kMod:
        break;
    }
    Opcode opcode = kAdd;
    switch (node.kind) {
      case SymExpr::Kind::kAdd:
        opcode = kAdd;
        break;
      case SymExpr::Kind::kMul:
        opcode = kMul;
        break;
      case SymExpr::Kind::kMax:
        opcode = kMax;
        break;
      case SymExpr::Kind::kMin:
        opcode = kMin;
        break;
      case SymExpr::Kind::kPow:
        opcode = kPow;
        break;
      case SymExpr::Kind::kFloorDiv:
        opcode = kFloorDiv;
        break;
      case SymExpr::Kind::kMod:
        opcode = kMod;
        break;
      case SymExpr::Kind::kConst:
      case SymExpr::Kind::kSymbol:
        break;
    }
    emit(node.args[0]);
    for (size_t i = 1; i < node.args.size(); ++i) {
      emit(node.args[i]);
      code.push_back(word(opcode));
      --depth;
    }
  };
  emit(expr);
  code.push_back(word(kEnd));
  return begin;
}

void DeviceSymbols::resolveSources(ExecutionState& state) {
  for (auto& [node, step, where] : state.symbolSources) {
    if (node < 0) {
      continue;
    }
    const auto& sv = state.stepVectors.at(node).at(step);
    const bool dims = (where & kDimsSource) != 0;
    where &= ~kDimsSource;
    const auto kernel = static_cast<size_t>(where >> 32);
    const auto index = static_cast<size_t>(where & 0xffffffff);
    where = sv.paramOffsets.at(kernel) +
        sv.kernels.at(kernel).returnOffsets.at(index);
    if (dims) {
      // dims[0] of the returned Tensor, an int32.
      where =
          (where + static_cast<int64_t>(offsetof(Tensor, dims))) | kDimsSource;
    }
  }
}

void DeviceSymbols::bindFixups(
    const ExecutionState& state,
    StepProgram& program) {
  if (state.substitutedSymbol.empty()) {
    return;
  }
  auto fixups = std::make_shared<SymbolFixups>();
  folly::F14FastSet<int32_t> picked;
  for (const auto& [id, index] : state.substitutedSymbol) {
    picked.insert(index);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto isPicked = [&](const std::string& name) {
    auto it = symbolIndex_.find(name);
    return it != symbolIndex_.end() && picked.count(it->second) != 0;
  };
  enum class Use { kHost, kPicked, kMixed };
  auto useOf = [&](const SymExpr& expr) {
    std::vector<std::string> names;
    expr.collectSymbols(names);
    const bool anyPicked = std::any_of(names.begin(), names.end(), isPicked);
    const bool allPicked = std::all_of(names.begin(), names.end(), isPicked);
    return !anyPicked ? Use::kHost : allPicked ? Use::kPicked : Use::kMixed;
  };
  for (const auto& fill : program.fills) {
    const SymExpr* start = nullptr;
    const bool view = fill.kind == FillOp::kTensor &&
        state.viewStarts.count(fill.valueId) != 0;
    if (view) {
      start = &state.viewStarts.at(fill.valueId);
    } else if (
        fill.kind == FillOp::kTensor &&
        state.bandStarts.count(fill.valueId) != 0) {
      start = &state.bandStarts.at(fill.valueId);
    }
    int32_t startCode = kHostValue;
    if (start != nullptr) {
      const auto startUse = useOf(*start);
      TORCH_CHECK(
          startUse != Use::kMixed,
          "Device symbols: a start mixes picked and host symbols: %",
          fill.valueId,
          " = ",
          start->toString());
      // A view starts from 0 on the host even when no picked symbol moves it.
      if (startUse == Use::kPicked || view) {
        startCode = emitProgram(*start, fixups->code);
      }
    }
    if (state.symbolRegion.count(fill.valueId) == 0 &&
        startCode == kHostValue) {
      continue;
    }
    if (fill.kind == FillOp::kScalar) {
      const auto* scalar = scalarOf(fill.valueId);
      if (scalar == nullptr) {
        continue;
      }
      const auto use = useOf(*scalar);
      if (use == Use::kHost) {
        continue;
      }
      TORCH_CHECK(
          use == Use::kPicked,
          "Device symbols: a scalar mixes picked and host symbols: %",
          fill.valueId,
          " = ",
          scalar->toString());
      std::vector<int32_t> record(kRecordWords, 0);
      record[0] = fill.offset;
      record[1] = kScalarRecord;
      record[4] = emitProgram(*scalar, fixups->code);
      fixups->records.insert(
          fixups->records.end(), record.begin(), record.end());
      ++fixups->numRecords;
      continue;
    }
    if (fill.kind != FillOp::kTensor && fill.kind != FillOp::kShapeOnly) {
      continue;
    }
    const bool shapeOnly = fill.kind == FillOp::kShapeOnly;
    const auto* shape = shapeOf(fill.valueId);
    std::vector<std::string> symbols;
    if (shape != nullptr) {
      symbolsOf(*shape, symbols);
    }
    if (shape == nullptr || shape->dims.size() > kMaxDims ||
        shape->dims.size() != shape->strides.size() ||
        std::none_of(symbols.begin(), symbols.end(), isPicked)) {
      TORCH_CHECK(
          !view,
          "Device symbols: a slice viewing its whole base has no picked dims: %",
          fill.valueId);
      if (startCode != kHostValue) {
        std::vector<int32_t> record(kRecordWords, 0);
        record[0] = fill.offset;
        record[1] = kBandRecord;
        record[3] = startCode;
        fixups->records.insert(
            fixups->records.end(), record.begin(), record.end());
        ++fixups->numRecords;
      }
      continue;
    }
    std::vector<int32_t> record(kRecordWords, 0);
    record[0] = fill.offset;
    record[1] = shapeOnly ? kShapeRecord : view ? kViewRecord : kTensorRecord;
    record[2] = static_cast<int32_t>(shape->dims.size());
    record[3] = startCode;
    for (size_t d = 0; d < shape->dims.size(); ++d) {
      const auto dimUse = useOf(shape->dims[d]);
      TORCH_CHECK(
          dimUse != Use::kMixed,
          "Device symbols: a dim mixes picked and host symbols: %",
          fill.valueId,
          " dim ",
          d,
          " = ",
          shape->dims[d].toString());
      record[4 + d] = dimUse == Use::kPicked
          ? emitProgram(shape->dims[d], fixups->code)
          : kHostValue;
      // A stride free of picked symbols keeps the host's value, which knows
      // layouts the export does not, such as a concat band. One that also
      // needs a host symbol is the dense stride of the real dims.
      const auto strideUse = useOf(shape->strides[d]);
      if (shapeOnly || strideUse == Use::kHost) {
        record[8 + d] = kHostValue;
      } else if (strideUse == Use::kMixed) {
        record[8 + d] = kDenseStride;
      } else {
        record[8 + d] = emitProgram(shape->strides[d], fixups->code);
      }
    }
    fixups->records.insert(fixups->records.end(), record.begin(), record.end());
    ++fixups->numRecords;
  }
  if (fixups->numRecords > 0) {
    program.symbolFixups = std::move(fixups);
  }
}

void DeviceSymbols::beginExecution(ExecutionState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  state.symbolCapacities.assign(symbols_.size(), 0);
  std::vector<std::pair<nativert::ValueId, int64_t>> values;
  for (const auto& [id, index] : state.substitutedSymbol) {
    const auto capacity = capacity_.at(symbols_.at(index));
    state.symbolCapacities.at(index) = capacity;
    if (candidateTensor_.count(id) == 0) {
      values.emplace_back(id, capacity);
    }
  }
  for (auto id : state.capacityScalars) {
    auto value = CHECK_NOTNULL(scalarOf(id))->evaluate(capacity_);
    TORCH_CHECK(value.has_value(), "Device symbols: no capacity for %", id);
    values.emplace_back(id, *value);
  }
  for (const auto& [id, capacity] : values) {
    writeFrame(*state.frame, id, c10::IValue(capacity));
    if (state.tableOnlyActive &&
        static_cast<size_t>(id) < state.tableOnly.size() &&
        state.tableOnly[id] != 0) {
      auto& entry = state.valueEntries.at(id);
      entry.scalar = capacity;
      entry.kind = ValueEntry::kInt;
      entry.generation = state.tableGeneration;
    }
  }
}

bool DeviceSymbols::observeReturnDims(
    ExecutionState& state,
    nativert::ValueId id,
    int64_t count) {
  if (state.substitutedSymbol.count(id) == 0) {
    return false;
  }
  auto it = candidateTensor_.find(id);
  if (it == candidateTensor_.end()) {
    return false;
  }
  auto [observed, inserted] =
      state.observedSymbols.try_emplace(it->second, count);
  if (!inserted && observed->second != count) {
    state.symbolConflicts.insert(it->second);
    observed->second = std::max(observed->second, count);
  }
  return true;
}

bool DeviceSymbols::observeReturn(
    ExecutionState& state,
    nativert::ValueId id,
    int64_t value) {
  if (state.capacityScalars.count(id) != 0) {
    return true;
  }
  auto it = candidateSymbol_.find(id);
  if (it == candidateSymbol_.end()) {
    return false;
  }
  auto [observed, inserted] =
      state.observedSymbols.try_emplace(it->second, value);
  if (!inserted && observed->second != value) {
    state.symbolConflicts.insert(it->second);
    observed->second = std::max(observed->second, value);
  }
  return state.substitutedSymbol.count(id) != 0;
}

bool DeviceSymbols::endExecution(ExecutionState& state) {
  bool fits = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Carriers merged as equal returned different values: a broadcast or
    // expand the merge read as an equality met a dim of 1. The symbol is not
    // picked again, and states bound with it are dropped.
    bool usedConflict = false;
    for (const auto& symbol : state.symbolConflicts) {
      if (banned_.insert(symbol).second) {
        LOG(WARNING) << "Device symbol " << symbol
                     << " has carriers with different values; not picked "
                        "again";
        ++generation_;
      }
      auto indexIt = symbolIndex_.find(symbol);
      for (const auto& [id, index] : state.substitutedSymbol) {
        if (indexIt != symbolIndex_.end() && index == indexIt->second) {
          usedConflict = true;
        }
      }
    }
    state.symbolConflicts.clear();
    if (usedConflict) {
      fits = false;
    }
    for (const auto& [id, index] : state.substitutedSymbol) {
      auto it = state.observedSymbols.find(symbols_[index]);
      if (it != state.observedSymbols.end() &&
          it->second > state.symbolCapacities.at(index)) {
        LOG(WARNING) << "Device symbol " << symbols_[index] << " = "
                     << it->second << " exceeded its capacity "
                     << state.symbolCapacities.at(index)
                     << "; the execution reruns";
        fits = false;
      }
    }
    constexpr int64_t kRound = 16;
    for (const auto& [symbol, value] : state.observedSymbols) {
      if (boundOf_.count(symbol) != 0) {
        continue;
      }
      const auto rounded = (value + kRound - 1) / kRound * kRound;
      auto [it, inserted] = capacity_.try_emplace(symbol, rounded);
      if (!inserted) {
        it->second = std::max(it->second, rounded);
      }
    }
  }
  // The graph outputs sized from capacities take their real dims. The
  // kernels wrote them with the real strides from the fixups.
  for (auto id : state.symbolOutputs) {
    const auto* shape = shapeOf(id);
    auto& frame = *state.frame;
    const auto& ivalue = frame.getIValue(id);
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    for (size_t d = 0; d < shape->dims.size(); ++d) {
      auto dim = shape->dims[d].evaluate(state.observedSymbols);
      auto stride = shape->strides[d].evaluate(state.observedSymbols);
      TORCH_CHECK(
          dim.has_value() && stride.has_value(),
          "Device symbols: no real dims for output %",
          id);
      dims.push_back(*dim);
      strides.push_back(*stride);
    }
    int64_t extent = tensor.storage_offset() + 1;
    for (size_t d = 0; d < dims.size(); ++d) {
      extent += (dims[d] - 1) * strides[d];
    }
    TORCH_CHECK(
        !fits || dims.empty() ||
            extent * tensor.element_size() <=
                static_cast<int64_t>(tensor.storage().nbytes()),
        "Device symbols: graph output %",
        id,
        " (",
        waveGraph_.idToValue().at(id)->name(),
        ") has real dims ",
        c10::IntArrayRef(dims),
        " past its buffer of ",
        tensor.storage().nbytes() / tensor.element_size(),
        " elements, sizes ",
        tensor.sizes());
    if (fits && tensor.sizes() != c10::IntArrayRef(dims)) {
      writeFrame(
          frame,
          id,
          c10::IValue(
              tensor.as_strided(dims, strides, tensor.storage_offset())));
    }
  }
  state.observedSymbols.clear();
  return fits;
}

FixupArgs DeviceSymbols::prepareFixups(
    ExecutionState& state,
    SymbolFixups& fixups) {
  if (!fixups.deviceRecords.defined()) {
    auto options = at::TensorOptions().device(at::kCUDA);
    fixups.deviceRecords = at::from_blob(
                               fixups.records.data(),
                               {static_cast<int64_t>(fixups.records.size())},
                               at::TensorOptions().dtype(at::kInt))
                               .to(options.dtype(at::kInt));
    fixups.deviceCode =
        at::from_blob(
            fixups.code.data(),
            {static_cast<int64_t>(std::max<size_t>(fixups.code.size(), 1))},
            at::TensorOptions().dtype(at::kLong))
            .to(options.dtype(at::kLong));
  }
  int32_t numSymbols = static_cast<int32_t>(state.symbolSources.size());
  std::vector<int64_t> symbols(2 * static_cast<size_t>(numSymbols), 0);
  for (int32_t i = 0; i < numSymbols; ++i) {
    const auto& [node, step, offset] = state.symbolSources[i];
    symbols[numSymbols + i] = state.symbolCapacities.at(i);
    if (node >= 0) {
      // An odd address marks an int32 source.
      symbols[i] = reinterpret_cast<int64_t>(
                       state.deviceBuffers.at(node).at(step)->as<uint8_t>() +
                       (offset & ~kDimsSource)) |
          ((offset & kDimsSource) != 0 ? 1 : 0);
    }
  }
  // The addresses change only when a step's parameter buffer does, the
  // capacities only between executions, so the copy is rare.
  if (symbols != state.symbolArgsHost) {
    state.symbolArgsHost = symbols;
    state.symbolArgsDevice =
        at::from_blob(
            symbols.data(),
            {static_cast<int64_t>(std::max<size_t>(symbols.size(), 1))},
            at::TensorOptions().dtype(at::kLong))
            .to(at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
  }
  return {
      .records = fixups.deviceRecords.data_ptr<int32_t>(),
      .code = fixups.deviceCode.data_ptr<int64_t>(),
      .numRecords = fixups.numRecords,
      .symbols = state.symbolArgsDevice.data_ptr<int64_t>(),
      .numSymbols = numSymbols,
      .numCode = static_cast<int32_t>(fixups.code.size())};
}

void DeviceSymbols::runFixups(
    ExecutionState& state,
    SymbolFixups& fixups,
    uint8_t* deviceBase,
    facebook::velox::wave::Stream* stream) {
  auto args = prepareFixups(state, fixups);
  void* kernelArgs[] = {
      &deviceBase,
      &args.records,
      &args.code,
      &args.numRecords,
      &args.symbols,
      &args.numSymbols,
      &args.numCode};
  fixupKernel().launch(
      0, 1, 256, fixupSharedBytes(args.numCode), stream, kernelArgs);
}

int32_t DeviceSymbols::inferReserveShapes(const ExecutionState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (WaveConfig::get().sizeExprsFromInputs) {
    return deriveShapesFromInputs(state);
  }
  return applySymReserves(state, *shapes_);
}

namespace {

// Makes each dim of 'shape', the shape of 'name', that has a fixed value a
// symbol of its own. Keeps the strides dense if they were.
void unfixDims(const std::string& name, SymShape& shape) {
  const auto dense = denseShape(shape.dims);
  bool contiguous = shape.strides.size() == dense.strides.size();
  for (size_t d = 0; contiguous && d < shape.strides.size(); ++d) {
    contiguous = sameDim(shape.strides.at(d), dense.strides.at(d));
  }
  for (size_t d = 0; d < shape.dims.size(); ++d) {
    auto& dim = shape.dims.at(d);
    if (const auto traced = tracedValue(dim)) {
      dim = hostDimSymbol(name, d, *traced);
    }
  }
  if (contiguous) {
    shape.strides = denseShape(shape.dims).strides;
  }
}

} // namespace

int32_t DeviceSymbols::deriveShapesFromInputs(const ExecutionState& state) {
  // Graph-input dims are symbols bound each execution; weights keep the
  // export's shapes. Everything else is derived from those through the shape
  // rules and the reserves' symbolic forms, ignoring the export's constants,
  // which hold the shapes the export was traced with.
  auto derived = std::make_shared<SymShapes>();
  derived->scalarByName = shapes_->scalarByName;
  derived->exported = shapes_.get();
  const auto& graph = *waveGraph_.graph();
  // The constant elements of int lists are values without a producer.
  for (const auto& [id, constant] : graph.getConstantSymIntValues()) {
    auto it = waveGraph_.idToValue().find(id);
    if (it != waveGraph_.idToValue().end()) {
      derived->scalarByName.insert_or_assign(
          std::string(it->second->name()), SymExpr::constant(constant));
    }
  }
  auto seed = [&](const nativert::Value* value) {
    const std::string name(value->name());
    auto it = shapes_->byName.find(name);
    if (it == shapes_->byName.end()) {
      return;
    }
    // A dim the export left symbolic keeps its expression. A dim it fixed
    // at c > 1 becomes symbol d<c>, shared by all input dims traced at c, so
    // that dims traced equal stay equal.
    std::vector<SymExpr> dims;
    for (const auto& dim : it->second.dims) {
      if (!dim.isConstant() || dim.value == 1) {
        dims.push_back(dim);
      } else {
        dims.push_back(
            SymExpr{
                .kind = SymExpr::Kind::kSymbol,
                .symbol = fmt::format("d{}", dim.value)});
      }
    }
    derived->byName.emplace(name, denseShape(std::move(dims)));
  };
  for (const auto* input : graph.userInputs()) {
    if (input == nullptr) {
      continue;
    }
    if (input->type().kind() == nativert::Type::Kind::Tensor) {
      seed(input);
      continue;
    }
    // A list input's elements are its unpacks' outputs.
    for (const auto* user : input->users()) {
      if (user->target() == "prim.ListUnpack") {
        for (const auto* output : user->outputs()) {
          seed(output);
        }
      }
    }
  }
  for (const auto* weight : graph.weightValues()) {
    auto it = shapes_->byName.find(std::string(weight->name()));
    if (it != shapes_->byName.end()) {
      derived->byName.emplace(it->first, it->second);
    }
  }
  int32_t numAdded = 0;
  auto converge = [&]() {
    for (int32_t round = 0; round < 8; ++round) {
      const auto added = inferSymShapes(*derived, waveGraph_) +
          applySymReserves(state, *derived) + shapeDuplicates(*derived) +
          shapeVariantNodes(*derived);
      numAdded += added;
      if (added == 0) {
        break;
      }
    }
  };
  converge();
  // An integer that is still unknown is one only the data decides, such as a
  // count a kernel returns: it becomes a symbol of its own, u_<value>, and
  // the shapes that depend on it follow.
  int32_t numMinted = 0;
  for (const auto& [id, value] : waveGraph_.idToValue()) {
    if (value->type().kind() == nativert::Type::Kind::SymInt &&
        value->producer() != nullptr &&
        derived->scalarByName
            .try_emplace(
                std::string(value->name()),
                SymExpr{
                    .kind = SymExpr::Kind::kSymbol,
                    .symbol = "u_" + std::string(value->name())})
            .second) {
      ++numMinted;
    }
  }
  if (numMinted > 0) {
    converge();
  }
  // An op no rule describes, such as an eager op that runs on the host, gets
  // the export's shapes for its outputs once its inputs have shapes: the
  // host decides those shapes when the op returns.
  auto hasShape = [&](const nativert::Value* value) {
    if (value == nullptr) {
      return true;
    }
    switch (value->type().kind()) {
      case nativert::Type::Kind::Tensor:
        return derived->byName.count(std::string(value->name())) != 0;
      case nativert::Type::Kind::TensorList: {
        const auto elements = listElementShapes(*derived, value);
        return !elements.empty() &&
            std::all_of(elements.begin(), elements.end(), [](auto* element) {
              return element != nullptr;
            });
      }
      case nativert::Type::Kind::None:
      case nativert::Type::Kind::NestedTensorList:
      case nativert::Type::Kind::OptionalTensorList:
      case nativert::Type::Kind::SymInt:
      case nativert::Type::Kind::SymIntList:
      case nativert::Type::Kind::SymBool:
      case nativert::Type::Kind::SymFloat:
      case nativert::Type::Kind::CustomObj:
        break;
    }
    return true;
  };
  // A dim the export fixed for a custom op is what the op's fake kernel made
  // at trace time, which the real op need not return: it becomes a symbol of
  // its own, which the setup reads from the value's shape slot. An aten op's
  // meta function gives the real dims, as it does for the tw op replacing it.
  auto fromExport = [&](const nativert::Node& producer,
                        const nativert::Value* value) {
    auto it = shapes_->byName.find(std::string(value->name()));
    if (it == shapes_->byName.end()) {
      return 0;
    }
    auto shape = it->second;
    if (!producer.target().starts_with("torch.ops.aten.") &&
        !producer.target().starts_with("tw.")) {
      VLOG(1) << "Device symbols: dims of " << it->first << " from "
              << producer.target() << " are the host's";
      unfixDims(it->first, shape);
    }
    if (!derived->byName.emplace(it->first, std::move(shape)).second) {
      return 0;
    }
    hostShaped_.insert(it->first);
    return 1;
  };
  for (int32_t round = 0; round < 4; ++round) {
    int32_t numFromExport = 0;
    folly::F14FastSet<const nativert::Node*> seen;
    for (const auto& [id, value] : waveGraph_.idToValue()) {
      const auto* node = value->producer();
      if (node == nullptr || node->target() == "tw.placeholder" ||
          node->target() == "prim.Input" || !seen.insert(node).second ||
          !std::all_of(
              node->inputs().begin(), node->inputs().end(), [&](auto& in) {
                return hasShape(in.value);
              })) {
        continue;
      }
      for (const auto* output : node->outputs()) {
        if (output == nullptr || hasShape(output)) {
          continue;
        }
        if (output->type().kind() == nativert::Type::Kind::Tensor) {
          numFromExport += fromExport(*node, output);
          continue;
        }
        for (const auto* user : output->users()) {
          if (user->target() == "prim.ListUnpack") {
            for (const auto* element : user->outputs()) {
              numFromExport += fromExport(*node, element);
            }
          }
        }
      }
    }
    if (numFromExport == 0) {
      break;
    }
    converge();
  }
  derivedShapes_ = std::move(derived);
  return numAdded;
}

int32_t DeviceSymbols::shapeVariantNodes(SymShapes& shapes) {
  const auto& idToValue = waveGraph_.idToValue();
  // The value of the main graph a variant's value stands for: the one with
  // its id. Where the names differ, what is known carries across.
  auto mainOf = [&](const nativert::Value* value) -> const nativert::Value* {
    auto it = idToValue.find(value->id());
    return it != idToValue.end() && it->second->name() != value->name()
        ? it->second
        : nullptr;
  };
  auto carry = [&](const nativert::Value* from, const nativert::Value* to) {
    auto shape = shapes.byName.find(std::string(from->name()));
    if (shape != shapes.byName.end()) {
      auto copy = shape->second;
      return shapes.byName.emplace(std::string(to->name()), std::move(copy))
                 .second
          ? 1
          : 0;
    }
    auto scalar = shapes.scalarByName.find(std::string(from->name()));
    if (scalar != shapes.scalarByName.end()) {
      auto copy = scalar->second;
      return shapes.scalarByName
                 .emplace(std::string(to->name()), std::move(copy))
                 .second
          ? 1
          : 0;
    }
    return 0;
  };
  int32_t numAdded = 0;
  for (const auto& graph : waveGraph_.variantGraphs()) {
    for (const auto& node : graph->nodes()) {
      for (const auto& input : node.inputs()) {
        if (input.value == nullptr) {
          continue;
        }
        if (const auto* main = mainOf(input.value)) {
          carry(main, input.value);
        }
      }
      inferNodeShapes(shapes, node);
      for (const auto* output : node.outputs()) {
        if (output == nullptr) {
          continue;
        }
        if (const auto* main = mainOf(output)) {
          numAdded += carry(output, main);
        } else if (idToValue.count(output->id()) != 0) {
          // Same name: the rule wrote the main value's shape directly.
        }
      }
    }
  }
  return numAdded;
}

int32_t DeviceSymbols::shapeDuplicates(SymShapes& shapes) {
  const auto& idToValue = waveGraph_.idToValue();
  auto valueOf = [&](int32_t id) -> const nativert::Value* {
    auto it = idToValue.find(id);
    return it != idToValue.end() ? it->second : nullptr;
  };
  int32_t numAdded = 0;
  for (const auto& [duplicateId, duplicate] : waveGraph_.duplicates()) {
    const auto* value = valueOf(duplicateId);
    const auto* original = valueOf(duplicate.original);
    const auto* node = original != nullptr ? original->producer() : nullptr;
    if (value == nullptr || node == nullptr ||
        shapes.byName.count(std::string(value->name())) != 0) {
      continue;
    }
    const auto& bindings = *duplicate.bindings;
    auto actualOf = [&](const nativert::Value* formal) {
      auto it = bindings.find(formal->id());
      return it != bindings.end() ? valueOf(it->second) : formal;
    };
    // The original's node, run over the invocation's actual operands: their
    // shapes stand in for the formal ones, and the outputs are cleared, for
    // as long as the rule runs.
    std::vector<std::pair<std::string, std::optional<SymShape>>> savedShapes;
    std::vector<std::pair<std::string, std::optional<SymExpr>>> savedScalars;
    auto saveShape = [&](const std::string& name) {
      auto it = shapes.byName.find(name);
      savedShapes.emplace_back(
          name,
          it != shapes.byName.end() ? std::optional<SymShape>(it->second)
                                    : std::nullopt);
    };
    bool known = true;
    for (const auto& input : node->inputs()) {
      const auto* formal = input.value;
      const auto* actual = formal != nullptr ? actualOf(formal) : nullptr;
      if (formal == nullptr || actual == nullptr || actual == formal) {
        continue;
      }
      const std::string formalName(formal->name());
      if (formal->type().kind() == nativert::Type::Kind::Tensor) {
        auto it = shapes.byName.find(std::string(actual->name()));
        if (it == shapes.byName.end()) {
          known = false;
          break;
        }
        auto shape = it->second;
        saveShape(formalName);
        shapes.byName.insert_or_assign(formalName, std::move(shape));
      } else if (formal->type().kind() == nativert::Type::Kind::SymInt) {
        auto it = shapes.scalarByName.find(std::string(actual->name()));
        if (it == shapes.scalarByName.end()) {
          known = false;
          break;
        }
        auto scalar = it->second;
        auto old = shapes.scalarByName.find(formalName);
        savedScalars.emplace_back(
            formalName,
            old != shapes.scalarByName.end()
                ? std::optional<SymExpr>(old->second)
                : std::nullopt);
        shapes.scalarByName.insert_or_assign(formalName, std::move(scalar));
      }
    }
    // A list output's rule writes its elements under listElementName.
    auto isList = [](const nativert::Value* output) {
      return output->type().kind() == nativert::Type::Kind::TensorList;
    };
    auto elementNames = [&](const nativert::Value* output) {
      std::vector<std::string> names;
      for (size_t i = 0;; ++i) {
        auto name = listElementName(output->name(), i);
        if (shapes.byName.count(name) == 0) {
          return names;
        }
        names.push_back(std::move(name));
      }
    };
    if (known) {
      for (const auto* output : node->outputs()) {
        if (output == nullptr) {
          continue;
        }
        saveShape(std::string(output->name()));
        shapes.byName.erase(std::string(output->name()));
        if (isList(output)) {
          for (const auto& name : elementNames(output)) {
            saveShape(name);
            shapes.byName.erase(name);
          }
        }
      }
      inferNodeShapes(shapes, *node);
      // The restore runs in reverse, so these are erased before the saved
      // elements come back.
      for (const auto* output : node->outputs()) {
        if (output != nullptr && isList(output)) {
          for (auto& name : elementNames(output)) {
            savedShapes.emplace_back(std::move(name), std::nullopt);
          }
        }
      }
    }
    std::vector<std::pair<std::string, SymShape>> results;
    if (known) {
      for (const auto* output : node->outputs()) {
        const auto* actual = output != nullptr ? actualOf(output) : nullptr;
        auto it = output != nullptr
            ? shapes.byName.find(std::string(output->name()))
            : shapes.byName.end();
        if (actual != nullptr && actual != output &&
            it != shapes.byName.end()) {
          results.emplace_back(std::string(actual->name()), it->second);
        }
      }
    }
    for (auto it = savedShapes.rbegin(); it != savedShapes.rend(); ++it) {
      if (it->second.has_value()) {
        shapes.byName.insert_or_assign(it->first, *it->second);
      } else {
        shapes.byName.erase(it->first);
      }
    }
    for (auto it = savedScalars.rbegin(); it != savedScalars.rend(); ++it) {
      if (it->second.has_value()) {
        shapes.scalarByName.insert_or_assign(it->first, *it->second);
      } else {
        shapes.scalarByName.erase(it->first);
      }
    }
    for (auto& [name, shape] : results) {
      if (shapes.byName.emplace(name, std::move(shape)).second) {
        ++numAdded;
      }
    }
  }
  return numAdded;
}

int32_t DeviceSymbols::applySymReserves(
    const ExecutionState& state,
    SymShapes& shapes) {
  const auto& idToValue = waveGraph_.idToValue();
  auto shapeIn = [&](int32_t id) -> const SymShape* {
    auto it = idToValue.find(id);
    if (it == idToValue.end()) {
      return nullptr;
    }
    auto found = shapes.byName.find(std::string(it->second->name()));
    return found != shapes.byName.end() ? &found->second : nullptr;
  };
  auto valueOf = [&](int32_t id) -> const nativert::Value* {
    auto it = idToValue.find(id);
    return it != idToValue.end() ? it->second : nullptr;
  };
  auto scalarExpr = [&](const ScalarRef& ref) -> std::optional<SymExpr> {
    if (ref.kind == ScalarRef::kConstant) {
      return SymExpr::constant(ref.constant);
    }
    const auto* value =
        ref.kind == ScalarRef::kValue ? valueOf(ref.valueId) : nullptr;
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::string name(value->name());
    auto it = shapes.scalarByName.find(name);
    if (it != shapes.scalarByName.end()) {
      return it->second;
    }
    SymExpr symbol{.kind = SymExpr::Kind::kSymbol, .symbol = "u_" + name};
    shapes.scalarByName.emplace(name, symbol);
    return symbol;
  };
  int32_t numAdded = 0;
  // A step's operands come from earlier steps; a second pass catches the
  // rest of any order the steps are listed in.
  for (int32_t pass = 0; pass < 2; ++pass) {
    for (const auto& steps : state.stepVectors) {
      for (const auto& sv : steps) {
        if (sv.program == nullptr) {
          continue;
        }
        // Each size's symbolic result, then each output it shapes: one
        // per element of a list, or several outputs sharing one size.
        const auto& sizes = sv.program->sizes;
        std::vector<std::optional<std::vector<SymShape>>> results(sizes.size());
        for (size_t s = 0; s < sizes.size(); ++s) {
          const auto& size = sizes[s];
          if (size.fn == nullptr || size.concat >= 0) {
            continue;
          }
          const auto sym = symReserveOf(size.fn);
          if (sym == nullptr) {
            continue;
          }
          SymReserveArgs args;
          args.blockSize = size.blockSize;
          for (const auto& group : size.args.tensors) {
            auto& shapesOfGroup = args.tensors.emplace_back();
            for (auto id : group) {
              shapesOfGroup.push_back(shapeIn(id));
            }
          }
          for (const auto& ref : size.args.scalars) {
            args.scalars.push_back(scalarExpr(ref));
          }
          bool listsKnown = true;
          for (const auto& list : size.args.intLists) {
            auto& exprs = args.intLists.emplace_back();
            for (const auto& ref : list) {
              auto expr = scalarExpr(ref);
              if (!expr.has_value()) {
                listsKnown = false;
                break;
              }
              exprs.push_back(std::move(*expr));
            }
          }
          std::vector<SymShape> out(std::max(1, size.numShapes));
          if (listsKnown && sym(args, out, static_cast<int32_t>(out.size()))) {
            results[s] = std::move(out);
          }
        }
        // The operands of a fused concat get their shapes from their own
        // sizes.
        for (const auto& concat : sv.program->concats) {
          for (size_t i = 0;
               i < concat.operandIds.size() && i < concat.operandSizes.size();
               ++i) {
            const auto s = concat.operandSizes[i];
            const auto* value = valueOf(concat.operandIds[i]);
            if (value == nullptr || s < 0 ||
                static_cast<size_t>(s) >= results.size() ||
                !results[s].has_value() || results[s]->empty()) {
              continue;
            }
            auto shape = (*results[s])[0];
            if (shapes.byName
                    .emplace(std::string(value->name()), std::move(shape))
                    .second) {
              ++numAdded;
            }
          }
        }
        for (const auto& launch : sv.program->launches) {
          for (const auto& bound : launch.largestOutputs) {
            const auto* value = valueOf(bound.valueId);
            if (value == nullptr || bound.size < 0 ||
                static_cast<size_t>(bound.size) >= results.size() ||
                !results[bound.size].has_value() ||
                static_cast<size_t>(bound.element) >=
                    results[bound.size]->size()) {
              continue;
            }
            auto shape = (*results[bound.size])[bound.element];
            if (shapes.byName
                    .emplace(std::string(value->name()), std::move(shape))
                    .second) {
              ++numAdded;
            }
          }
          for (const auto& bound : launch.outputs) {
            const auto* value = valueOf(bound.valueId);
            if (value == nullptr) {
              continue;
            }
            const std::string name(value->name());
            if (bound.size >= 0 &&
                static_cast<size_t>(bound.size) < sizes.size()) {
              const auto& size = sizes[bound.size];
              if (size.data != nullptr && size.outputIdx >= 0 &&
                  static_cast<size_t>(size.outputIdx) <
                      size.data->actualOutputDescs.size() &&
                  size.data->actualOutputDescs[size.outputIdx]
                      .shapeSetOnDevice) {
                setOnDevice_.insert(name);
              }
            }
            const SymShape* shape = nullptr;
            if (bound.kind == BoundOutput::kAlias && bound.aliasOf >= 0) {
              shape = shapeIn(bound.aliasOf);
            } else if (
                (bound.kind == BoundOutput::kTensor ||
                 bound.kind == BoundOutput::kListElement ||
                 bound.kind == BoundOutput::kShapeOnly) &&
                bound.size >= 0 &&
                static_cast<size_t>(bound.size) < results.size() &&
                results[bound.size].has_value() &&
                static_cast<size_t>(bound.element) <
                    results[bound.size]->size()) {
              shape = &(*results[bound.size])[bound.element];
            }
            if (shape != nullptr) {
              auto copy = *shape;
              if (shapes.byName.emplace(name, std::move(copy)).second) {
                ++numAdded;
              }
            }
          }
        }
      }
    }
  }
  return numAdded;
}

namespace {

// The value's name and its producer's target, for messages.
std::string describeValue(const nativert::Value* value) {
  if (value == nullptr) {
    return "unknown value";
  }
  const auto* producer = value->producer();
  return fmt::format(
      "{} from {}",
      value->name(),
      producer != nullptr ? producer->target() : "input");
}

} // namespace

namespace {

// True if every symbol of 'expr' is one only the data decides.
bool onlyDataDependent(const SymExpr& expr) {
  std::vector<std::string> names;
  expr.collectSymbols(names);
  return !names.empty() &&
      std::all_of(names.begin(), names.end(), [](const auto& name) {
        return name[0] == 'u';
      });
}

} // namespace

void DeviceSymbols::checkSizeExprs(
    ExecutionState& state,
    const StepProgram& program,
    const uint8_t* params,
    int32_t sequenceNumber,
    int32_t stepIdx) const {
  auto& symbols = state.sizeExprSymbols;
  auto producerOf = [&](int32_t id) {
    auto it = waveGraph_.idToValue().find(id);
    return describeValue(
        it != waveGraph_.idToValue().end() ? it->second : nullptr);
  };
  // The producer's target, and for a list element the list's producer's.
  auto producerKey = [&](int32_t id) {
    auto it = waveGraph_.idToValue().find(id);
    const auto* producer =
        it != waveGraph_.idToValue().end() ? it->second->producer() : nullptr;
    if (producer == nullptr) {
      return std::string("input");
    }
    std::string key(producer->target());
    if (producer->target() == "prim.ListUnpack" &&
        !producer->inputs().empty() && producer->inputs()[0].value != nullptr) {
      const auto* list = producer->inputs()[0].value->producer();
      key += "<-";
      key += list != nullptr ? std::string(list->target()) : "input";
    }
    return key;
  };
  // Integer parameters carry the actual value of a data-dependent symbol;
  // the step's own outputs may hold a capacity for it.
  if (derivedShapes_ != nullptr) {
    for (const auto& fill : program.fills) {
      if (fill.kind != FillOp::kScalar) {
        continue;
      }
      auto it = waveGraph_.idToValue().find(fill.valueId);
      if (it == waveGraph_.idToValue().end()) {
        continue;
      }
      auto scalar =
          derivedShapes_->scalarByName.find(std::string(it->second->name()));
      if (scalar != derivedShapes_->scalarByName.end() &&
          scalar->second.kind == SymExpr::Kind::kSymbol) {
        symbols.try_emplace(
            scalar->second.symbol,
            *reinterpret_cast<const int64_t*>(params + fill.offset));
      }
    }
  }
  folly::F14FastSet<int32_t> outputs;
  for (const auto& launch : program.launches) {
    for (const auto& bound : launch.outputs) {
      outputs.insert(bound.valueId);
    }
  }
  for (const auto& fill : program.fills) {
    if (fill.kind != FillOp::kTensor && fill.kind != FillOp::kShapeOnly) {
      continue;
    }
    const auto* tensor = reinterpret_cast<const Tensor*>(params + fill.offset);
    if (fill.kind == FillOp::kTensor && tensor->storage == nullptr) {
      continue;
    }
    ++numSizeValues_;
    const SymShape* shape = nullptr;
    if (derivedShapes_ != nullptr) {
      auto it = waveGraph_.idToValue().find(fill.valueId);
      if (it != waveGraph_.idToValue().end()) {
        auto found =
            derivedShapes_->byName.find(std::string(it->second->name()));
        shape =
            found != derivedShapes_->byName.end() ? &found->second : nullptr;
      }
    } else {
      shape = shapeOf(fill.valueId);
    }
    if (shape != nullptr && isHostShaped(fill.valueId)) {
      ++numSizeHostShaped_;
      continue;
    }
    // The host keeps a 0-dim tensor as one element of rank 1.
    if (shape != nullptr && shape->dims.empty() && tensor->rank == 1 &&
        tensor->dims[0] == 1) {
      ++numSizeMatched_;
      continue;
    }
    if (shape == nullptr ||
        shape->dims.size() != static_cast<size_t>(tensor->rank)) {
      ++numSizeNoShape_;
      auto key = producerKey(fill.valueId);
      std::lock_guard<std::mutex> lock(noShapeMutex_);
      ++noShapeByTarget_[shape != nullptr && key == "input" ? "rank" : key];
      continue;
    }
    // An output whose extent is a symbol only the data decides holds a
    // capacity here; it is compared once the step's consumers bind the
    // symbol, at the end of the execution.
    if (outputs.count(fill.valueId) != 0) {
      std::vector<std::string> names;
      for (const auto& dim : shape->dims) {
        dim.collectSymbols(names);
      }
      if (std::any_of(names.begin(), names.end(), [&](const auto& name) {
            return !name.empty() && name[0] == 'u' && symbols.count(name) == 0;
          })) {
        std::lock_guard<std::mutex> lock(noShapeMutex_);
        pendingSizeChecks_.push_back(
            {.valueId = fill.valueId,
             .dims = shape->dims,
             .hostDims = std::vector<int64_t>(
                 tensor->dims, tensor->dims + tensor->rank)});
        continue;
      }
    }
    bool differ = false;
    bool unbound = false;
    bool setOnDevice = false;
    for (int32_t d = 0; d < tensor->rank; ++d) {
      const auto& expr = shape->dims[d];
      const auto value = expr.evaluate(symbols);
      if (!value.has_value()) {
        // A device-set shape is a capacity in its producer's frame.
        if (expr.kind == SymExpr::Kind::kSymbol &&
            isSetOnDevice(fill.valueId)) {
          setOnDevice = true;
        } else if (expr.kind == SymExpr::Kind::kSymbol) {
          symbols[expr.symbol] = tensor->dims[d];
        } else {
          unbound = true;
          if (numSizeUnboundReports_++ < 10) {
            LOG(WARNING) << "Size expression of %" << fill.valueId << " dim "
                         << d << " at step " << sequenceNumber << "." << stepIdx
                         << " has an unbound symbol: " << expr.toString();
          }
        }
        continue;
      }
      if (*value != tensor->dims[d]) {
        // A device-set shape, or a capacity for an extent only the data
        // decides: the host holds such a symbol's capacity in some steps and
        // its actual value in others.
        if (isSetOnDevice(fill.valueId) ||
            (*value < tensor->dims[d] && outputs.count(fill.valueId) != 0) ||
            (*value > tensor->dims[d] && onlyDataDependent(expr))) {
          setOnDevice = true;
          continue;
        }
        differ = true;
        if (numSizeReports_++ < 20) {
          LOG(WARNING) << "Size expression of %" << fill.valueId << " dim " << d
                       << " at step " << sequenceNumber << "." << stepIdx
                       << " gives " << *value << " = " << expr.toString()
                       << ", the host sized " << tensor->dims[d] << "; "
                       << producerOf(fill.valueId);
        }
      }
    }
    if (differ) {
      std::lock_guard<std::mutex> lock(noShapeMutex_);
      ++differByTarget_[producerKey(fill.valueId)];
    }
    ++(differ            ? numSizeDiffer_
           : unbound     ? numSizeUnbound_
           : setOnDevice ? numSizeSetOnDevice_
                         : numSizeMatched_);
  }
}

bool DeviceSymbols::isHostShaped(int32_t valueId) const {
  auto it = waveGraph_.idToValue().find(valueId);
  return it != waveGraph_.idToValue().end() &&
      hostShaped_.count(std::string(it->second->name())) != 0;
}

bool DeviceSymbols::isSetOnDevice(int32_t valueId) const {
  auto it = waveGraph_.idToValue().find(valueId);
  return it != waveGraph_.idToValue().end() &&
      setOnDevice_.count(std::string(it->second->name())) != 0;
}

void DeviceSymbols::reportSizeExprs(
    const folly::F14FastMap<std::string, int64_t>& symbols) const {
  {
    std::lock_guard<std::mutex> lock(noShapeMutex_);
    for (const auto& pending : pendingSizeChecks_) {
      bool unbound = false;
      bool differ = false;
      bool capacity = false;
      for (size_t d = 0; d < pending.dims.size(); ++d) {
        const auto value = pending.dims[d].evaluate(symbols);
        if (!value.has_value()) {
          unbound = true;
        } else if (*value < pending.hostDims[d]) {
          capacity = true;
        } else if (*value > pending.hostDims[d]) {
          differ = true;
        }
      }
      ++(differ         ? numSizeDiffer_
             : unbound  ? numSizeUnbound_
             : capacity ? numSizeSetOnDevice_
                        : numSizeMatched_);
    }
    pendingSizeChecks_.clear();
  }
  if (numSizeValues_ == 0) {
    return;
  }
  LOG(INFO) << "Size expressions: " << numSizeValues_.exchange(0)
            << " tensor parameters, " << numSizeMatched_.exchange(0)
            << " match, " << numSizeDiffer_.exchange(0) << " differ, "
            << numSizeUnbound_.exchange(0) << " unbound, "
            << numSizeNoShape_.exchange(0) << " without an expression, "
            << numSizeSetOnDevice_.exchange(0)
            << " differ in a shape the device sets or a capacity, "
            << numSizeHostShaped_.exchange(0) << " shaped by host ops";
  numSizeReports_ = 0;
  numSizeUnboundReports_ = 0;
  std::lock_guard<std::mutex> lock(noShapeMutex_);
  auto byProducer = [](folly::F14FastMap<std::string, int64_t>& counts) {
    std::vector<std::pair<int64_t, std::string>> byCount;
    for (const auto& [target, count] : counts) {
      byCount.emplace_back(count, target);
    }
    counts.clear();
    std::sort(byCount.rbegin(), byCount.rend());
    std::string text;
    for (size_t i = 0; i < byCount.size() && i < 20; ++i) {
      text += fmt::format(" {} x{}", byCount[i].second, byCount[i].first);
    }
    return text;
  };
  LOG(INFO) << "  without an expression, by producer:"
            << byProducer(noShapeByTarget_);
  LOG(INFO) << "  differ, by producer:" << byProducer(differByTarget_);
}

int32_t DeviceSymbols::fixupSharedBytes(int32_t numCode) {
  constexpr int32_t kSharedCodeWords = 2048;
  return numCode <= kSharedCodeWords
      ? numCode * static_cast<int32_t>(sizeof(int64_t))
      : 0;
}

const char* DeviceSymbols::fixupSource() {
  return kFixupSource;
}

void DeviceSymbols::traceWaits(
    const ExecutionState& state,
    const folly::F14FastSet<int32_t>& returned) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    const auto& steps = state.stepVectors[node];
    for (size_t step = 0; step < steps.size(); ++step) {
      const auto& program = steps[step].program;
      if (program == nullptr ||
          program->numIndependent >=
              static_cast<int32_t>(program->launches.size())) {
        continue;
      }
      std::string waits;
      for (auto id : program->waitReads) {
        if (returned.count(id) == 0) {
          continue;
        }
        const auto* value = waveGraph_.idToValue().at(id);
        const std::string* symbol = nullptr;
        for (const auto* candidates : {&candidateSymbol_, &candidateTensor_}) {
          auto it = candidates->find(id);
          if (it != candidates->end()) {
            symbol = &it->second;
          }
        }
        std::string why = "not a single data symbol";
        if (const auto* scalar = scalarOf(id)) {
          why = fmt::format(
              "= {}{}",
              scalar->toString(),
              state.symbolRegion.count(id) != 0 ? " in a region" : "");
        } else if (const auto* shape = shapeOf(id)) {
          why = "shape";
          for (const auto& dim : shape->dims) {
            why += " " + dim.toString();
          }
        }
        if (symbol != nullptr) {
          auto reasonIt = reason_.find(*symbol);
          why = *symbol + ": " +
              (reasonIt != reason_.end() ? reasonIt->second : "picked");
        }
        waits += fmt::format(
            " %{}({}; {})",
            id,
            value->producer() ? value->producer()->target() : "-",
            why);
      }
      LOG(INFO) << "Waiting step " << node << "." << step << ":" << waits;
    }
  }
}

std::string DeviceSymbols::report(const ExecutionState& state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string symbols;
  for (const auto& [id, index] : state.substitutedSymbol) {
    symbols += fmt::format(
        " {}=%{}<={}",
        symbols_[index],
        id,
        index < static_cast<int32_t>(state.symbolCapacities.size())
            ? state.symbolCapacities[index]
            : -1);
  }
  return fmt::format(
      "Device symbols: {} packed lists not returned; {} integers from "
      "symbols; {} of {} candidate values run on capacities:{}\n",
      state.numUnreturnedLists,
      state.capacityScalars.size(),
      state.substitutedSymbol.size(),
      candidateSymbol_.size(),
      symbols);
}

} // namespace torch::wave
