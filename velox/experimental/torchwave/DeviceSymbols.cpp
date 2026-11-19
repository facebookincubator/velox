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
#include <cstring>
#include <deque>
#include <functional>
#include <map>

#include <torch/nativert/executor/ExecutionFrame.h>
#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/DeviceGrid.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/StaticDims.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/Utils.h"
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
// Tensor and shape records that bindSizeChecks writes: their symbols have no
// capacity, so they keep their sizes when a device symbol is over.
constexpr int32_t kSizedTensorRecord = 7;
constexpr int32_t kSizedShapeRecord = 8;
// A record that checks a tensor parameter's dims; see fixupBody.
constexpr int32_t kCheckRecord = 5;
constexpr int32_t kCheckDataSymbols = 1;
constexpr int32_t kCheckShapeOnly = 2;
// The record's dims are the capacity the reserve gives an output its kernel
// shapes.
constexpr int32_t kCheckCapacity = 4;
// A record that checks the data pointer of a member of an allocation group
// against the program at rec[4], its group's base plus the bytes before it.
// rec[3] is as in a check record.
constexpr int32_t kPlacementRecord = 9;
// A record that writes the program at rec[4] to the data pointer at rec[0].
constexpr int32_t kPlacedRecord = 10;
// Records that place the members of an allocation group, run before the
// others: the program at rec[4] into temp rec[3]; an exclusive scan of the
// rec[4] temps from rec[3]; the program at rec[4] to the address whose low
// and high words are rec[5] and rec[6].
constexpr int32_t kTempRecord = 11;
constexpr int32_t kScanRecord = 12;
constexpr int32_t kStoreRecord = 13;
// Resets a word the kernels change: rec[2] is 0 for an int64 to zero, 1 for
// an int32 to zero, 2 for a tensor parameter's status.
constexpr int32_t kVolatileRecord = 14;
// After the scan of a group's temps: writes the group's total, temp rec[3], to
// its demand slot, the address whose low and high words are rec[5] and rec[6],
// and when the total exceeds the capacity, program rec[4], marks the step
// failed in the failure words (spec S1-S3).
constexpr int32_t kCapacityRecord = 15;
// Temps per step, as the fixup kernel has them.
constexpr int32_t kNumTemps = 1024;
// A symbol named this, then a temp's index, is that temp.
constexpr const char* kTempPrefix = "__temp_";
// A record that copies the value of symbol rec[3] to element rec[4] of the
// table at the address of symbol rec[5].
constexpr int32_t kCopyRecord = 6;
// Elements of the table that keeps tensor elements for later steps.
constexpr int32_t kTableSize = 4096;
// Dim and stride codes in a tensor record that are not program offsets.
constexpr int32_t kHostValue = -1;
constexpr int32_t kDenseStride = -2;

// Program words: the opcode in the top byte, an operand below it.
// A symbol named this, then a decimal device address, is the int64 at that
// address. It takes no slot in the symbols.
constexpr const char* kLoadPrefix = "__load_";
// A symbol named this prefix + a symbol's name is that symbol's capacity.
constexpr const char* kCapacityPrefix = "__cap_";
// Marks a symbol in a program's header whose capacity alone is read.
constexpr int64_t kCapacityOnly = int64_t{1} << 20;

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
  // Replaces the address on top of the stack with the int64 there.
  kLoad = 11,
  // Pushes the temp the operand names.
  kPushTemp = 12,
  // Pushes the capacity of the symbol the operand names.
  kPushCapacity = 13,
};

constexpr int64_t kOperandBits = 56;
constexpr int64_t kOperandMask = (int64_t{1} << kOperandBits) - 1;

int64_t word(Opcode opcode, int64_t operand = 0) {
  return (static_cast<int64_t>(opcode) << kOperandBits) |
      (operand & kOperandMask);
}

const char* const kFixupSource = R"(
constexpr int kMaxSymbols = 1024;
constexpr int kMaxTemps = 1024;
constexpr long long kCapacityOnly = 1LL << 20;

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

__device__ long long evalProgram(
    const long long* code,
    const long long* values,
    const long long* temps = nullptr,
    const long long* caps = nullptr) {
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
    if (op == 11) {
      stack[top - 1] = *reinterpret_cast<const long long*>(stack[top - 1]);
      continue;
    }
    if (op == 12) {
      stack[top++] = temps[operand];
      continue;
    }
    if (op == 13) {
      stack[top++] = caps[operand];
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
    int barrierId,
    const int* failure,
    int step) {
  const int tid = threadIdx.x - firstThread;
  // 'symbols' holds numSymbols source addresses, then their capacities.
  __shared__ long long values[kMaxSymbols];
  __shared__ long long capacities[kMaxSymbols];
  // All loads go before the one barrier: the programs, the symbols, and a
  // prefetch of each thread's first record's slot. Each word of a program in
  // global memory is otherwise a dependent load.
  const int firstRecord = tid < numRecords ? records[tid * 12] : -1;
  const long long* code_ = code;
  if (codeBuffer != nullptr) {
    for (int i = tid; i < numCode; i += numThreads) {
      codeBuffer[i] = code[i];
    }
    code = codeBuffer;
  }
  int anyOver = 0;
  // code[0] is how many symbols the programs read, code[1...] their indices.
  const long long* used = code_ + 1;
  const int numUsed = static_cast<int>(code_[0]);
  for (int j = tid; j < numUsed; j += numThreads) {
    const int i = static_cast<int>(used[j] & (kCapacityOnly - 1));
    const long long address = (used[j] & kCapacityOnly) ? 0 : symbols[i];
    const long long capacity = symbols[numSymbols + i];
    capacities[i] = capacity;
    if (used[j] & kCapacityOnly) {
      continue;
    }
    long long v = 0;
    if (address & 2) {
      // Element (address >> 48) of the integer tensor whose parameter is at
      // the address.
      const auto* slot = reinterpret_cast<const unsigned char*>(
          address & 0xfffffffffffcLL);
      const long long index = (address >> 48) & 0x7fff;
      const void* data = *reinterpret_cast<void* const*>(slot);
      v = slot[9] == 4 ? static_cast<const int*>(data)[index]
                       : static_cast<const long long*>(data)[index];

    } else if (address != 0) {
      v = (address & 1) ? *reinterpret_cast<const int*>(address - 1)
                        : *reinterpret_cast<const long long*>(address);
    }
    values[i] = v;
    anyOver |= v < 0 || v > capacity ? 1 : 0;
  }
  if (firstRecord >= 0) {
    asm volatile("prefetch.global.L1 [%0];" ::"l"(params + firstRecord));
  }
  FIXUP_MARK(0);
  const int over = barrierOr(barrierId, numThreads, anyOver);
  // The addresses of allocation group members: each member's aligned size
  // into 'temps', an exclusive scan per group, then each address to its
  // table slot, which the records below load.
  __shared__ long long temps[kMaxTemps];
  for (int r = tid; r < numRecords; r += numThreads) {
    const int* rec = records + r * 12;
    if (rec[1] == 11) {
      temps[rec[3]] = evalProgram(code + rec[4], values, temps, capacities);
    }
  }
  barrier(barrierId, numThreads);
  for (int r = tid; r < numRecords; r += numThreads) {
    const int* rec = records + r * 12;
    if (rec[1] == 12) {
      long long sum = 0;
      for (int i = rec[3]; i < rec[3] + rec[4]; ++i) {
        const long long size = temps[i];
        temps[i] = sum;
        sum += size;
      }
    }
  }
  barrier(barrierId, numThreads);
  for (int r = tid; r < numRecords; r += numThreads) {
    const int* rec = records + r * 12;
    if (rec[1] == 13 || rec[1] == 15) {
      auto* dest = reinterpret_cast<long long*>(
          static_cast<unsigned long long>(static_cast<unsigned>(rec[5])) |
          (static_cast<unsigned long long>(static_cast<unsigned>(rec[6])) << 32));
      if (rec[1] == 13) {
        *dest = evalProgram(code + rec[4], values, temps, capacities);
        continue;
      }
      // The setup runs to its end either way; the demand tells the host what
      // the group needs.
      const long long demand = temps[rec[3]];
      *dest = demand;
      if (failure != nullptr &&
          demand > evalProgram(code + rec[4], values, temps, capacities)) {
        atomicMin(const_cast<int*>(failure), step);
      }
    }
  }
  barrier(barrierId, numThreads);
  FIXUP_MARK(1);
  for (int r = tid; r < numRecords; r += numThreads) {
    const int* rec = records + r * 12;
    if (rec[1] >= 11 && rec[1] <= 13 || rec[1] == 15) {
      continue;
    }
    unsigned char* slot = params + rec[0];
    if (rec[1] == 14) {
      if (rec[2] == 0) {
        *reinterpret_cast<long long*>(slot) = 0;
      } else if (rec[2] == 1) {
        *reinterpret_cast<int*>(slot) = 0;
      } else {
        *reinterpret_cast<unsigned*>(slot + 48) = 0;
      }
      continue;
    }
    if (rec[1] == 1) {
      *reinterpret_cast<long long*>(slot) = over ? 0 : evalProgram(code + rec[4], values, nullptr, capacities);
      continue;
    }
    if (rec[1] == 6) {
      reinterpret_cast<long long*>(symbols[rec[5]])[rec[4]] = values[rec[3]];
      continue;
    }
    if (rec[1] == 9 || rec[1] == 10) {
      // rec[5], when set, is an offset in elements of the parameter's size.
      const long long device =
          evalProgram(code + rec[4], values, nullptr, capacities) +
          (rec[5] >= 0
               ? evalProgram(code + rec[5], values, nullptr, capacities) *
                   slot[9]
               : 0);
      if (rec[1] == 10) {
        *reinterpret_cast<long long*>(slot) = device;
        continue;
      }
      auto* counts =
          reinterpret_cast<unsigned long long*>(symbols[rec[3] & 0x3ff]);
      const long long host = *reinterpret_cast<const long long*>(slot);
      atomicAdd(counts + 8, 1ULL);
      if (host != device) {
        const auto n = atomicAdd(counts + 9, 1ULL);
        if (n < 10) {
          printf("Device size check: placement %%%d device %llx host %llx\n",
                 rec[3] >> 10, device, host);
        }
      }
      continue;
    }
    if (rec[1] == 5) {
      // A size check: the record's dims and strides against the host's.
      // rec[2] is the rank, with flags from bit 8: 1 data-decided symbols,
      // 2 shape only. rec[3] is the symbol whose address is the counts, with
      // the value id from bit 10. A stride coded -1 is not checked. The
      // counts: checked, dims differ, dims differ on a data-decided symbol,
      // rank differs, strides differ, strides differ on a data-decided
      // symbol.
      const int rank = rec[2] & 0xff;
      const int flags = rec[2] >> 8;
      auto* counts =
          reinterpret_cast<unsigned long long*>(symbols[rec[3] & 0x3ff]);
      if ((flags & 2) == 0 && *reinterpret_cast<void**>(slot) == nullptr) {
        continue;
      }
      if (static_cast<int>(static_cast<signed char>(slot[8])) != rank) {
        atomicAdd(counts + 3, 1ULL);
        continue;
      }
      const int* hostDims = reinterpret_cast<const int*>(slot + 12);
      const int* hostStrides = reinterpret_cast<const int*>(slot + 28);
      // An empty tensor's strides are never stepped.
      bool empty = false;
      for (int d = 0; d < rank; ++d) {
        empty |= hostDims[d] == 0;
      }
      int differ = -1;
      int strideDiffer = -1;
      long long device = 0;
      long long deviceStride = 0;
      for (int d = 0; d < rank; ++d) {
        const long long dim = evalProgram(code + rec[4 + d], values, nullptr, capacities);
        if (dim != hostDims[d] && differ < 0) {
          differ = d;
          device = dim;
        }
        // The host reads a size-1 dim through stride 0.
        if (rec[8 + d] >= 0 && hostDims[d] != 1 && !empty) {
          const long long stride = evalProgram(code + rec[8 + d], values, nullptr, capacities);
          if (stride != hostStrides[d] && strideDiffer < 0) {
            strideDiffer = d;
            deviceStride = stride;
          }
        }
      }
      if (flags & 4) {
        // A capacity check: dims only, counted apart.
        atomicAdd(counts + 6, 1ULL);
        if (differ >= 0) {
          const auto n = atomicAdd(counts + 7, 1ULL);
          if (n < 10) {
            printf("Device size check: capacity %%%d dim %d device %lld host %d\n",
                   rec[3] >> 10, differ, device, hostDims[differ]);
          }
        }
        continue;
      }
      atomicAdd(counts, 1ULL);
      if (differ >= 0) {
        const auto n = atomicAdd(counts + 1 + (flags & 1), 1ULL);
        if (n < 10) {
          printf("Device size check: %%%d dim %d device %lld host %d flags %d\n",
                 rec[3] >> 10, differ, device, hostDims[differ], flags);
        }
      } else if (strideDiffer >= 0) {
        const auto n = atomicAdd(counts + 4 + (flags & 1), 1ULL);
        if (n < 10) {
          printf("Device size check: %%%d stride %d device %lld host %d flags %d\n",
                 rec[3] >> 10, strideDiffer, deviceStride,
                 hostStrides[strideDiffer], flags);
        }
      }
      continue;
    }
    // Kinds 7 and 8 are tensor and shape records over symbols that have no
    // capacity: they keep their sizes when another symbol is over.
    const int kind = rec[1] == 7 ? 0 : rec[1] == 8 ? 2 : rec[1];
    const int recordOver = rec[1] >= 7 ? 0 : over;
    // A band the host placed at its start over the capacities moves to its
    // start over the real values.
    if (rec[3] >= 0 && !recordOver) {
      const long long move = evalProgram(code + rec[3], values, nullptr, capacities) -
          (kind == 4 ? 0 : evalProgram(code + rec[3], capacities));
      *reinterpret_cast<unsigned char**>(slot) += move * slot[9];
    }
    if (kind == 3) {
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
      const long long dim = recordOver ? 0
          : rec[4 + d] >= 0 ? evalProgram(code + rec[4 + d], values, nullptr, capacities)
                            : dims[d];
      dims[d] = static_cast<int>(dim);
      if (rec[8 + d] >= 0) {
        strides[d] = static_cast<int>(evalProgram(code + rec[8 + d], values, nullptr, capacities));
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
    if (kind != 2) {
      *reinterpret_cast<bool*>(slot + 52) = contiguous;
    }
  }
}

// True when a step at or before 'firstStep' failed: the setup returns, and
// the composite kernels that follow return at entry.
__device__ inline bool stepAlreadyFailed(const int* failure, int firstStep) {
  return failure != nullptr &&
      *reinterpret_cast<const volatile int*>(failure) < firstStep;
}

extern "C" __global__ void fixupSymbols(
    unsigned char* params,
    const int* records,
    const long long* code,
    int numRecords,
    const long long* symbols,
    int numSymbols,
    int numCode,
    const int* failure,
    int firstStep) {
  if (stepAlreadyFailed(failure, firstStep)) {
    return;
  }
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
      1,
      failure,
      firstStep);
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

// Marks a symbol source whose offset is an absolute device address.
constexpr int32_t kAddressSource = -2;
// Mark sources in bindSizeChecks that are a slot of the input symbols'
// buffer, and an element of the table.
constexpr int32_t kInputSource = -3;
constexpr int32_t kTableSource = -4;

// Marks a symbol source that is an element of an integer tensor: the offset
// of its parameter in the low 32 bits, the element index above them.
constexpr int64_t kElementSource = int64_t{1} << 61;

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
  return !name.empty() &&
      (name[0] == 'u' || parseElementSymbol(name).has_value());
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
        if (node.symbol.starts_with(kTempPrefix)) {
          push(word(
              kPushTemp,
              std::stoll(node.symbol.substr(std::strlen(kTempPrefix)))));
          return;
        }
        if (node.symbol.starts_with(kLoadPrefix)) {
          push(word(
              kPushConst,
              std::stoll(node.symbol.substr(std::strlen(kLoadPrefix)))));
          code.push_back(word(kLoad));
          return;
        }
        if (node.symbol.starts_with(kCapacityPrefix)) {
          push(word(
              kPushCapacity,
              symbolIndex_.at(
                  node.symbol.substr(std::strlen(kCapacityPrefix)))));
          return;
        }
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
  for (auto index : state.sizeCheckSymbols) {
    state.symbolCapacities.at(index) = std::numeric_limits<int64_t>::max();
  }
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

namespace {

// Puts ahead of the programs of 'fixups' the symbols they read, so the
// kernel loads only those: a symbol of another step may be an element of a
// tensor not yet allocated in this execution. Moves the records' program
// offsets past the header.
void addSymbolHeader(SymbolFixups& fixups) {
  std::vector<int64_t> used;
  folly::F14FastSet<int64_t> seen;
  for (const auto word : fixups.code) {
    if ((word >> 56) == kPushSymbol) {
      const auto index = word & ((int64_t{1} << 56) - 1);
      if (seen.insert(index).second) {
        used.push_back(index);
      }
    }
  }
  // A symbol whose capacity alone is read is not loaded: its source may be
  // a tensor not yet allocated.
  for (const auto word : fixups.code) {
    if ((word >> 56) == kPushCapacity) {
      const auto index = word & ((int64_t{1} << 56) - 1);
      if (seen.insert(index).second) {
        used.push_back(index | kCapacityOnly);
      }
    }
  }
  for (int32_t r = 0; r < fixups.numRecords; ++r) {
    const auto* record = fixups.records.data() + r * kRecordWords;
    if (record[1] == kCopyRecord && seen.insert(record[3]).second) {
      used.push_back(record[3]);
    }
  }
  const auto shift = static_cast<int32_t>(used.size() + 1);
  std::vector<int64_t> code;
  code.reserve(fixups.code.size() + shift);
  code.push_back(static_cast<int64_t>(used.size()));
  code.insert(code.end(), used.begin(), used.end());
  code.insert(code.end(), fixups.code.begin(), fixups.code.end());
  fixups.code = std::move(code);
  auto move = [&](int32_t& offset) {
    if (offset >= 0) {
      offset += shift;
    }
  };
  for (int32_t r = 0; r < fixups.numRecords; ++r) {
    auto* record = fixups.records.data() + r * kRecordWords;
    if (record[1] == kCopyRecord) {
      continue;
    }
    if (record[1] == kPlacementRecord || record[1] == kPlacedRecord) {
      move(record[4]);
      move(record[5]);
      continue;
    }
    if (record[1] == kTempRecord || record[1] == kStoreRecord ||
        record[1] == kCapacityRecord) {
      move(record[4]);
      continue;
    }
    if (record[1] == kScanRecord || record[1] == kVolatileRecord) {
      continue;
    }
    if (record[1] == kCheckRecord) {
      for (int32_t d = 0; d < (record[2] & 0xff); ++d) {
        move(record[4 + d]);
        move(record[8 + d]);
      }
      continue;
    }
    for (int32_t i = 3; i < kRecordWords; ++i) {
      move(record[i]);
    }
  }
}

} // namespace

FixupArgs DeviceSymbols::prepareFixups(
    ExecutionState& state,
    SymbolFixups& fixups) {
  if (!fixups.deviceRecords.defined()) {
    addSymbolHeader(fixups);
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
    if (node == kAddressSource) {
      symbols[i] = offset;
    } else if (node >= 0 && (offset & kElementSource) != 0) {
      // Bit 1 marks an element source, the index goes in the top bits.
      symbols[i] = reinterpret_cast<int64_t>(
                       state.deviceBuffers.at(node).at(step)->as<uint8_t>() +
                       (offset & 0xffffffff)) |
          2 | (((offset >> 32) & 0x7fff) << 48);
    } else if (node >= 0) {
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
    facebook::velox::wave::Stream* stream,
    bool checkFailure) {
  auto args = prepareFixups(state, fixups);
  const int* failure = checkFailure && state.failureWords.defined()
      ? state.failureWords.data_ptr<int32_t>()
      : nullptr;
  int firstStep = state.stepOrdinal;
  void* kernelArgs[] = {
      &deviceBase,
      &args.records,
      &args.code,
      &args.numRecords,
      &args.symbols,
      &args.numSymbols,
      &args.numCode,
      &failure,
      &firstStep};
  fixupKernel().launch(
      0, 1, 1024, fixupSharedBytes(args.numCode), stream, kernelArgs);
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
  int32_t numAdded = 0;
  derivedShapes_ = deriveOnce(state, {}, numAdded);
  // The bands of concats the host lays out have the strides of the result;
  // a second pass starts from them, so views of a band read the band's
  // strides.
  folly::F14FastMap<std::string, SymShape> bands;
  const auto& idToValue = waveGraph_.idToValue();
  auto shapeOf = [&](nativert::ValueId id) -> const SymShape* {
    auto it = idToValue.find(id);
    if (it == idToValue.end()) {
      return nullptr;
    }
    auto found = derivedShapes_->byName.find(std::string(it->second->name()));
    return found != derivedShapes_->byName.end() ? &found->second : nullptr;
  };
  for (const auto& concat : hostConcats(state)) {
    const auto* result = shapeOf(concat.resultId);
    if (result == nullptr || result->strides.size() != result->dims.size()) {
      continue;
    }
    for (const auto& operandBands : concat.bands) {
      for (auto band : operandBands) {
        const auto* shape = shapeOf(band);
        auto strides = result->strides;
        if (concat.stack && concat.dim < static_cast<int64_t>(strides.size())) {
          strides.erase(strides.begin() + concat.dim);
        }
        if (shape == nullptr || shape->dims.size() != strides.size()) {
          continue;
        }
        bands.insert_or_assign(
            std::string(idToValue.at(band)->name()),
            SymShape{.dims = shape->dims, .strides = std::move(strides)});
      }
    }
  }
  if (!bands.empty()) {
    derivedShapes_ = deriveOnce(state, bands, numAdded);
  }
  return numAdded;
}

std::shared_ptr<SymShapes> DeviceSymbols::deriveOnce(
    const ExecutionState& state,
    const folly::F14FastMap<std::string, SymShape>& preset,
    int32_t& numAdded) {
  // Graph-input dims are symbols bound each execution; weights keep the
  // export's shapes. Everything else is derived from those through the shape
  // rules and the reserves' symbolic forms, ignoring the export's constants,
  // which hold the shapes the export was traced with.
  capacityByName_.clear();
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
  for (const auto& [name, shape] : preset) {
    derived->byName.emplace(name, shape);
  }
  numAdded = 0;
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
  return derived;
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
        // A shape the device sets is the rule's, not the capacity the
        // reserve gives.
        for (const auto& launch : sv.program->launches) {
          for (const auto& bound : launch.outputs) {
            const auto* value = valueOf(bound.valueId);
            if (value == nullptr || bound.size < 0 ||
                static_cast<size_t>(bound.size) >= sizes.size()) {
              continue;
            }
            const auto& size = sizes[bound.size];
            if (size.data != nullptr && size.outputIdx >= 0 &&
                static_cast<size_t>(size.outputIdx) <
                    size.data->actualOutputDescs.size() &&
                size.data->actualOutputDescs[size.outputIdx].shapeSetOnDevice) {
              setOnDevice_.insert(std::string(value->name()));
            }
          }
        }
        auto setOnDevice = [&](const nativert::Value* value) {
          return setOnDevice_.count(std::string(value->name())) != 0;
        };
        for (const auto& concat : sv.program->concats) {
          for (size_t i = 0;
               i < concat.operandIds.size() && i < concat.operandSizes.size();
               ++i) {
            const auto s = concat.operandSizes[i];
            const auto* value = valueOf(concat.operandIds[i]);
            if (value == nullptr || setOnDevice(value) || s < 0 ||
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
            if (value == nullptr || setOnDevice(value) || bound.size < 0 ||
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
              capacityByName_.insert_or_assign(name, *shape);
            }
            if (shape != nullptr && !setOnDevice(value)) {
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
    if ((fill.kind != FillOp::kTensor && fill.kind != FillOp::kShapeOnly) ||
        fill.kernel == FillOp::kExport) {
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

namespace {

using Step = std::pair<int32_t, int32_t>;

// Where a symbol's value is on the device for the setup kernels: an offset
// in the parameters of step (node, step), or an absolute address with node
// kAddressSource. 'afterStep' marks a value a kernel of that step leaves,
// which only later steps can read.
struct SizeSource {
  int32_t node;
  int32_t step;
  int64_t offset;
  bool afterStep{false};
  // When set, the device reads the value at this absolute address; 'node'
  // and 'step' still say from when it is there.
  int64_t address{0};

  Step at() const {
    return {node, step};
  }
};

// The device sources bindSizeChecks finds for the symbols of the derived
// shapes.
struct SizeSources {
  SizeSources(const SymShapes& shapes, const IdToValueMap& idToValue)
      : shapes(shapes), idToValue(idToValue) {}

  std::string nameOf(int32_t id) const {
    auto it = idToValue.find(id);
    return it != idToValue.end() ? std::string(it->second->name())
                                 : std::string();
  }

  const SymShape* shapeOf(int32_t id) const {
    auto it = shapes.byName.find(nameOf(id));
    return it != shapes.byName.end() ? &it->second : nullptr;
  }

  // The symbol a value's integer is, or null.
  const std::string* scalarSymbolOf(int32_t id) const {
    auto it = shapes.scalarByName.find(nameOf(id));
    return it != shapes.scalarByName.end() &&
            it->second.kind == SymExpr::Kind::kSymbol
        ? &it->second.symbol
        : nullptr;
  }

  const SymShapes& shapes;
  const IdToValueMap& idToValue;
  folly::F14FastMap<std::string, SizeSource> sources;
  // By value name: the parameters of the steps whose kernels write it, in
  // execution order.
  folly::F14FastMap<std::string, std::vector<SizeSource>> producedAt;
  // By name of a graph input tensor, or of an element of an input list: the
  // slot of the input values where the host writes its address each
  // execution.
  folly::F14FastMap<std::string, int64_t> inputAddress;
};

// The offset of dim 'd' of the tensor parameter at 'slot', an int32 source.
int64_t dimSource(int32_t slot, size_t d) {
  return (slot + static_cast<int64_t>(offsetof(Tensor, dims)) +
          static_cast<int64_t>(d * sizeof(int32_t))) |
      kDimsSource;
}

// Gives each input symbol a slot of the buffer the host writes at each
// execution start: the dims of input tensors, and input integers. Records
// where each slot's value is in the frame in 'state.sizeCheckInputs'.
// The 'dim' of a size-check input that is the input's address.
constexpr int32_t kAddressDim = -2;

void placeInputSymbols(
    const nativert::Graph& graph,
    ExecutionState& state,
    SizeSources& out) {
  state.sizeCheckInputs.clear();
  auto add = [&](const SymExpr& expr,
                 int32_t valueId,
                 int32_t element,
                 int32_t dim) {
    // A d<c> symbol is an input dim too: the export traced it at c, and
    // other inputs may have another extent there.
    if (expr.kind != SymExpr::Kind::kSymbol) {
      return;
    }
    const auto slot = static_cast<int64_t>(state.sizeCheckInputs.size());
    if (out.sources.try_emplace(expr.symbol, SizeSource{kInputSource, 0, slot})
            .second) {
      state.sizeCheckInputs.emplace_back(valueId, element, dim);
    }
  };
  auto addValue =
      [&](const nativert::Value* value, int32_t valueId, int32_t element) {
        if (value->type().kind() == nativert::Type::Kind::Tensor) {
          const auto slot = static_cast<int64_t>(state.sizeCheckInputs.size());
          if (out.inputAddress.try_emplace(std::string(value->name()), slot)
                  .second) {
            state.sizeCheckInputs.emplace_back(valueId, element, kAddressDim);
          }
          if (const auto* shape = out.shapeOf(value->id())) {
            for (size_t d = 0; d < shape->dims.size(); ++d) {
              add(shape->dims[d], valueId, element, static_cast<int32_t>(d));
            }
          }
          return;
        }
        auto it = out.shapes.scalarByName.find(std::string(value->name()));
        if (it != out.shapes.scalarByName.end()) {
          add(it->second, valueId, element, -1);
        }
      };
  for (const auto* input : graph.inputs()) {
    if (input == nullptr) {
      continue;
    }
    addValue(input, input->id(), -1);
    for (const auto* user : input->users()) {
      if (user->target() == "prim.ListUnpack") {
        for (size_t i = 0; i < user->outputs().size(); ++i) {
          addValue(user->outputs()[i], input->id(), static_cast<int32_t>(i));
        }
      }
    }
  }
}

// The values eager ops produce, list elements included: the host has their
// dims and values once the op returns.
folly::F14FastSet<int32_t> hostDecidedValues(
    const ExecutionState& state,
    const IdToValueMap& idToValue) {
  folly::F14FastSet<int32_t> result;
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      for (const auto& data : sv.standalones) {
        for (auto id : data.actualOutputs) {
          result.insert(id);
          auto it = idToValue.find(id);
          if (it == idToValue.end()) {
            continue;
          }
          for (const auto* user : it->second->users()) {
            if (user->target() == "prim.ListUnpack") {
              for (const auto* element : user->outputs()) {
                result.insert(element->id());
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// Adds the sources the kernels of one step leave: returned counts, dims of
// outputs whose shape the kernel sets, and the parameters of the outputs
// whose elements a size reads.
void addKernelSources(
    const StepVectors& sv,
    Step here,
    const folly::F14FastMap<int32_t, int32_t>& slotOf,
    SizeSources& out) {
  for (size_t k = 0; k < sv.kernels.size(); ++k) {
    const auto& data = sv.kernels[k];
    for (size_t r = 0; r < data.returnValues.size(); ++r) {
      const auto* symbol = data.returnTypes[r] == nativert::Type::Kind::SymInt
          ? out.scalarSymbolOf(data.returnValues[r])
          : nullptr;
      if (symbol != nullptr && isDataSymbol(*symbol)) {
        out.sources.try_emplace(
            *symbol,
            SizeSource{
                here.first,
                here.second,
                sv.paramOffsets.at(k) + data.returnOffsets.at(r),
                true});
      }
    }
    for (size_t i = 0; i < data.actualOutputs.size(); ++i) {
      auto slot = slotOf.find(data.actualOutputs[i]);
      if (slot == slotOf.end()) {
        continue;
      }
      out.producedAt[out.nameOf(data.actualOutputs[i])].push_back(
          SizeSource{here.first, here.second, slot->second, true});
      const auto* shape = out.shapeOf(data.actualOutputs[i]);
      if (i >= data.actualOutputDescs.size() ||
          !data.actualOutputDescs[i].shapeSetOnDevice || shape == nullptr ||
          shape->dims.size() > kMaxDims) {
        continue;
      }
      for (size_t d = 0; d < shape->dims.size(); ++d) {
        const auto& dim = shape->dims[d];
        if (dim.kind == SymExpr::Kind::kSymbol && isDataSymbol(dim.symbol)) {
          out.sources.try_emplace(
              dim.symbol,
              SizeSource{
                  here.first, here.second, dimSource(slot->second, d), true});
        }
      }
    }
  }
}

// Adds the sources the host writes in one step's parameters for values of
// eager ops: integers, and the dims of tensors.
void addHostSources(
    const StepProgram& program,
    Step here,
    const folly::F14FastSet<int32_t>& hostDecided,
    SizeSources& out) {
  for (const auto& fill : program.fills) {
    if (hostDecided.count(fill.valueId) == 0 ||
        fill.kernel == FillOp::kExport) {
      continue;
    }
    if (fill.kind == FillOp::kScalar) {
      if (const auto* symbol = out.scalarSymbolOf(fill.valueId)) {
        out.sources.try_emplace(
            *symbol, SizeSource{here.first, here.second, fill.offset});
      }
      continue;
    }
    const auto* shape =
        fill.kind == FillOp::kTensor ? out.shapeOf(fill.valueId) : nullptr;
    if (shape == nullptr || shape->dims.size() > kMaxDims) {
      continue;
    }
    for (size_t d = 0; d < shape->dims.size(); ++d) {
      const auto& dim = shape->dims[d];
      if (dim.kind == SymExpr::Kind::kSymbol &&
          (isHostDimSymbol(dim.symbol) || !tracedValue(dim))) {
        out.sources.try_emplace(
            dim.symbol,
            SizeSource{here.first, here.second, dimSource(fill.offset, d)});
      }
    }
  }
}

// Finds the sources of the symbols kernels and eager ops decide, in
// execution order, so each symbol keeps its first source.
void findStepSources(const ExecutionState& state, SizeSources& out) {
  const auto hostDecided = hostDecidedValues(state, out.idToValue);
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      const auto& sv = state.stepVectors[node][step];
      if (sv.program == nullptr) {
        continue;
      }
      const Step here{static_cast<int32_t>(node), static_cast<int32_t>(step)};
      folly::F14FastMap<int32_t, int32_t> slotOf;
      for (const auto& fill : sv.program->fills) {
        if ((fill.kind == FillOp::kTensor || fill.kind == FillOp::kShapeOnly) &&
            fill.kernel != FillOp::kExport) {
          slotOf.try_emplace(fill.valueId, fill.offset);
        }
      }
      addKernelSources(sv, here, slotOf, out);
      addHostSources(*sv.program, here, hostDecided, out);
    }
  }
}

// Elements of tensors that sizes read. An element is read in the setup of
// the step after the last one that writes its tensor, while the tensor still
// holds it, and kept in a table for the steps after that.
struct ElementTable {
  explicit ElementTable(const ExecutionState& state) {
    for (size_t node = 0; node < state.stepVectors.size(); ++node) {
      for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
        // A step without kernels runs no setup.
        const auto& sv = state.stepVectors[node][step];
        if (sv.program != nullptr && !sv.kernels.empty()) {
          setupSteps.emplace_back(node, step);
        }
      }
    }
  }

  // Gives the element symbols among 'names' sources for a record of step
  // 'here', and renames in 'dims' those read from the table. Updates
  // 'names'.
  void route(
      Step here,
      std::vector<SymExpr>& dims,
      std::vector<std::string>& names,
      SizeSources& out) {
    folly::F14FastMap<std::string, SymExpr> toTable;
    for (const auto& name : names) {
      auto element = parseElementSymbol(name);
      auto writers =
          element ? out.producedAt.find(element->first) : out.producedAt.end();
      if (writers == out.producedAt.end() || element->second >= 0x8000) {
        continue;
      }
      const SizeSource* last = nullptr;
      for (const auto& writer : writers->second) {
        if (writer.at() < here) {
          last = &writer;
        }
      }
      if (last == nullptr) {
        continue;
      }
      auto source = *last;
      source.offset |= kElementSource | (element->second << 32);
      // The symbol has one source for all steps, so a step whose last
      // writer differs goes unchecked.
      auto [it, inserted] = out.sources.try_emplace(name, source);
      if (!inserted && it->second.offset != source.offset) {
        it->second.node = std::numeric_limits<int32_t>::max();
      }
      // The writer's value is copied into the table at the next setup step.
      // With none after the writer, this step copies it.
      const auto next =
          std::upper_bound(setupSteps.begin(), setupSteps.end(), last->at());
      const auto copyStep = next == setupSteps.end() ? here : *next;
      if (copyStep == here) {
        continue;
      }
      auto [slot, added] =
          slots.try_emplace(name, static_cast<int32_t>(slots.size()));
      if (slot->second >= kTableSize) {
        continue;
      }
      if (added) {
        copies[copyStep].push_back(name);
      }
      const auto kept = name + "@t";
      out.sources.try_emplace(kept, SizeSource{kTableSource, 0, slot->second});
      toTable[name] = SymExpr{.kind = SymExpr::Kind::kSymbol, .symbol = kept};
    }
    if (toTable.empty()) {
      return;
    }
    names.clear();
    for (auto& dim : dims) {
      substituteSymbols(dim, toTable);
      dim.collectSymbols(names);
    }
  }

  std::vector<Step> setupSteps;
  // By step: the elements its setup copies to the table.
  std::map<Step, std::vector<std::string>> copies;
  // By element symbol: its slot in the table.
  folly::F14FastMap<std::string, int32_t> slots;
};

// ' name xcount' for the 'limit' largest counts.
std::string largestCounts(
    const folly::F14FastMap<std::string, int64_t>& counts,
    size_t limit) {
  std::vector<std::pair<int64_t, std::string>> byCount;
  for (const auto& [name, count] : counts) {
    byCount.emplace_back(count, name);
  }
  std::sort(byCount.rbegin(), byCount.rend());
  std::string text;
  for (size_t i = 0; i < byCount.size() && i < limit; ++i) {
    text += fmt::format(" {} x{}", byCount.at(i).second, byCount.at(i).first);
  }
  return text;
}

at::Tensor deviceLongs(int64_t size) {
  return at::zeros(
      {std::max<int64_t>(1, size)},
      at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
}

int64_t addressOf(const at::Tensor& tensor, int64_t index = 0) {
  return reinterpret_cast<int64_t>(tensor.data_ptr<int64_t>() + index);
}

// The values of a group: its members, and for a concat group the result and
// what is written into its bands.
std::vector<int32_t> groupValues(const AllocGroup& group) {
  std::vector<int32_t> ids(group.actualIds.begin(), group.actualIds.end());
  if (group.concat != nullptr) {
    ids.push_back(group.concat->resultId);
    for (const auto& operand : group.concat->operands) {
      ids.push_back(operand.writtenValueId());
      ids.push_back(operand.copyDestId);
    }
  }
  ids.erase(
      std::remove_if(ids.begin(), ids.end(), [](auto id) { return id < 0; }),
      ids.end());
  return ids;
}

// Puts on the device path the steps of 'candidates' (fully set up, every
// group's demand from the setup, every output in a group, no output the host
// reads) that also need no host work from the steps around them: no step off
// the path reads their outputs, and each group is carved and produced into
// by steps on the path only. Marks those steps and their values.
void selectDeviceAllocSteps(
    ExecutionState& state,
    const folly::F14FastMap<int32_t, const AllocGroupPlan*>& planOf,
    const std::vector<std::pair<int32_t, int32_t>>& candidates,
    const std::function<const SymShape*(int32_t)>& shapeOf,
    const std::function<std::optional<ExecutionState::HostFetchShape>(int32_t)>&
        hostShapeOf,
    const std::function<std::string(int32_t)>& producerOf_) {
  // The setup reads the shape or address of such a value from a slot the
  // host writes at its fill. A concat result's address slot the host writes
  // when it allocates the group.
  folly::F14FastSet<int32_t> concatResults;
  for (const auto& [node, plan] : planOf) {
    for (const auto& group : plan->groups) {
      if (group.concat != nullptr) {
        concatResults.insert(group.concat->resultId);
      }
    }
  }
  auto isHostWritten = [&](int32_t id) {
    return (id < static_cast<int32_t>(state.valueShapeSlot.size()) &&
            state.valueShapeSlot[id] >= 0) ||
        (id < static_cast<int32_t>(state.hasValueAddress.size()) &&
         state.hasValueAddress[id] != 0 && concatResults.count(id) == 0);
  };
  folly::F14FastMap<int32_t, int64_t> producerOf;
  // The step of each tensor a launch writes, bound or not: a kernel variant's
  // stages write values the step's program does not bind.
  folly::F14FastMap<int32_t, int64_t> writerOf;
  // The base of each view the steps make.
  folly::F14FastMap<int32_t, int32_t> viewBaseOf;
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      const auto& sv = state.stepVectors[node][step];
      if (sv.program == nullptr) {
        continue;
      }
      const auto key =
          stepKey(static_cast<int32_t>(node), static_cast<int32_t>(step));
      for (const auto& data : sv.kernels) {
        for (auto id : data.actualOutputs) {
          writerOf.emplace(id, key);
        }
      }
      for (const auto& launch : sv.program->launches) {
        for (const auto* outputs : {&launch.outputs, &launch.largestOutputs}) {
          for (const auto& bound : *outputs) {
            producerOf.emplace(bound.valueId, key);
            if (bound.kind == BoundOutput::kView && bound.derive >= 0) {
              const auto& code = sv.program->code;
              const auto op = code.viewOf(code.viewDerives[bound.derive]);
              if (!op.operands.empty() && op.operands[0] >= 0) {
                viewBaseOf.emplace(bound.valueId, op.operands[0]);
              }
            }
          }
        }
      }
      for (const auto& derive : sv.program->derives) {
        producerOf.emplace(derive.output, key);
        for (auto id : derive.outputs) {
          producerOf.emplace(id, key);
        }
      }
      for (const auto& derive : sv.program->derives) {
        if (derive.kind == StandaloneShortcut::kListPack ||
            derive.kind == StandaloneShortcut::kSymSize ||
            derive.kind == StandaloneShortcut::kSymNumel) {
          continue;
        }
        const auto base = derive.operands.empty() ? -1 : derive.operands[0];
        if (derive.output >= 0 && base >= 0) {
          viewBaseOf.emplace(derive.output, base);
        }
        for (size_t i = 0; i < derive.outputs.size(); ++i) {
          const auto source = derive.kind == StandaloneShortcut::kListUnpack &&
                  i < derive.elements.size()
              ? derive.elements[i]
              : base;
          if (source >= 0) {
            viewBaseOf.emplace(derive.outputs[i], source);
          }
        }
      }
    }
  }
  // Where the producer's parameters hold each kernel output's descriptor,
  // which a step off the path can fetch.
  folly::F14FastMap<int32_t, int32_t> descriptorOffset;
  state.fetchHostShapes.clear();
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      const auto& sv = state.stepVectors[node][step];
      if (sv.program == nullptr) {
        continue;
      }
      const auto key =
          stepKey(static_cast<int32_t>(node), static_cast<int32_t>(step));
      folly::F14FastSet<int32_t> outputs;
      for (const auto& launch : sv.program->launches) {
        for (const auto* list : {&launch.outputs, &launch.largestOutputs}) {
          for (const auto& bound : *list) {
            outputs.insert(bound.valueId);
          }
        }
      }
      // A returned value's dims come with its return, after the kernel; the
      // descriptor the setup wrote has its capacity.
      for (const auto& data : sv.kernels) {
        for (auto id : data.returnValues) {
          outputs.erase(id);
        }
      }
      // A member of device symbols' regions has its real dims in the
      // descriptor, where the host's derives see the capacities: the host
      // gives it those when it can evaluate them.
      for (const auto& fill : sv.program->fills) {
        auto it = producerOf.find(fill.valueId);
        // A kernel output, or a view the setup writes in an export slot.
        const bool exported = fill.kernel == FillOp::kExport &&
            sv.program->placedExports.count(fill.offset) != 0;
        if (fill.kind != FillOp::kTensor ||
            (outputs.count(fill.valueId) == 0 && !exported) ||
            it == producerOf.end() || it->second != key) {
          continue;
        }
        if (state.symbolRegion.count(fill.valueId) != 0) {
          auto hostShape = hostShapeOf(fill.valueId);
          if (!hostShape) {
            continue;
          }
          state.fetchHostShapes[fill.valueId] = std::move(*hostShape);
        }
        descriptorOffset.emplace(fill.valueId, fill.offset);
      }
    }
  }
  // The static header of a path value's parameter: rank and element type,
  // which the setup does not write.
  auto staticHeader =
      [&](int32_t id) -> std::optional<std::pair<int8_t, c10::ScalarType>> {
    const auto* types = state.valueTypes;
    if (types == nullptr || id < 0 ||
        static_cast<size_t>(id) >= types->types.size() ||
        static_cast<size_t>(id) >= types->constraints.size() ||
        types->types[id] == nullptr || types->constraints[id].rank < 0) {
      return std::nullopt;
    }
    return std::make_pair(
        types->constraints[id].rank, types->types[id]->dtype());
  };
  auto isFrameRead = [&](int32_t id) {
    return id >= 0 && id < static_cast<int32_t>(state.tableOnly.size()) &&
        state.tableOnly[id] == 0;
  };
  // Which values lack a static header, by producer and by what is missing.
  folly::F14FastMap<std::string, int64_t> noHeaderWhy;
  auto noteNoHeader = [&](int32_t id) {
    const auto* types = state.valueTypes;
    const bool noType = types == nullptr || id < 0 ||
        static_cast<size_t>(id) >= types->types.size() ||
        types->types[id] == nullptr;
    ++noHeaderWhy[fmt::format(
        "{} ({})", producerOf_(id), noType ? "no type" : "rank -1")];
  };
  folly::F14FastSet<int64_t> onPath;
  folly::F14FastMap<std::string, int64_t> why;
  // Why each step is off the path, for the reasons of the steps it drops.
  folly::F14FastMap<int64_t, std::string> offWhy;
  for (const auto& [node, step] : candidates) {
    offWhy[stepKey(node, step)] = "dropped";
  }
  for (const auto& [node, step] : candidates) {
    const auto& sv = state.stepVectors.at(node).at(step);
    const auto& program = *sv.program;
    // A metadata-only standalone that makes sizes, not tensors, reads the
    // capacities like the rest of the host, and an empty factory needs no
    // size. An eager one computes on real sizes the path does not fetch, and
    // a host view needs its base in the frame.
    bool eagerStandalone = false;
    for (const auto* list : {&sv.standalones, &sv.shortcutStandalones}) {
      for (const auto& data : *list) {
        const auto* launch = data.launch;
        bool makesTensor = false;
        if (launch != nullptr && launch->standalone != nullptr) {
          for (const auto* output : launch->standalone->outputs()) {
            makesTensor = makesTensor ||
                (output != nullptr &&
                 (output->type().kind() == nativert::Type::Kind::Tensor ||
                  output->type().kind() == nativert::Type::Kind::TensorList));
          }
        }
        eagerStandalone = eagerStandalone || launch == nullptr ||
            !launch->metadataOnly ||
            (makesTensor &&
             launch->standaloneShortcut != StandaloneShortcut::kEmptyFactory);
      }
    }
    // A path step without a device grid would send the whole execution to
    // the host path, so the static reasons maybePackDeviceGrid refuses on
    // keep a step off the path here.
    const char* reason = program.launches.empty() ? "no launches"
        : !deviceGridFits(program.grid, static_cast<int32_t>(sv.kernels.size()))
        ? "too many launches for the device grid"
        : sv.segments.size() > 1 ? "split into segments"
        : eagerStandalone        ? "standalones"
        : !sv.hostCopies.empty() ? "host copies"
                                 : nullptr;
    // The derives do not run: the setups of the steps that read their
    // outputs place them.
    // A host reader fetches a derived tensor from its export slot.
    for (const auto& derive : program.derives) {
      auto unfetchable = [&](int32_t id) {
        return descriptorOffset.count(id) == 0;
      };
      bool hostRead = (derive.materialize || isFrameRead(derive.output)) &&
          unfetchable(derive.output);
      for (size_t i = 0; i < derive.outputs.size(); ++i) {
        hostRead = hostRead ||
            (((i < derive.materializeOutputs.size() &&
               derive.materializeOutputs[i]) ||
              isFrameRead(derive.outputs[i])) &&
             unfetchable(derive.outputs[i]));
      }
      if (reason == nullptr && hostRead) {
        reason = "derived value the host reads";
      }
      // A sym_size or sym_numel is a scalar: no tensor header, its readers
      // get it from a scalar record.
      const bool scalarOutput = derive.kind == StandaloneShortcut::kSymSize ||
          derive.kind == StandaloneShortcut::kSymNumel;
      if (reason == nullptr && derive.output >= 0 && !scalarOutput &&
          !staticHeader(derive.output)) {
        reason = "derived value without a static rank and dtype";
        noteNoHeader(derive.output);
      }
    }
    for (const auto& launch : program.launches) {
      for (const auto* outputs : {&launch.outputs, &launch.largestOutputs}) {
        for (const auto& bound : *outputs) {
          // A view over the table is placed by the setups that read it, and
          // a host reader fetches it from its export slot; one over the
          // frame needs the host.
          const bool tableView =
              bound.kind == BoundOutput::kView && bound.derive >= 0;
          // A cat or stack laid out over the table: the path allocates the
          // group its result is in, and the setup places its bands.
          const bool tableConcat = bound.kind == BoundOutput::kConcat &&
              bound.size >= 0 && program.code.sizes[bound.size].concat >= 0;
          if (reason == nullptr && tableView && isFrameRead(bound.valueId) &&
              descriptorOffset.count(bound.valueId) == 0) {
            reason = "view output the host reads";
          } else if (
              reason == nullptr && bound.kind != BoundOutput::kTensor &&
              !tableView && !tableConcat) {
            reason = bound.kind == BoundOutput::kView
                ? "view output over the frame"
                : bound.kind == BoundOutput::kListElement
                ? "list element output"
                : bound.kind == BoundOutput::kShapeOnly ? "shape-only output"
                : bound.kind == BoundOutput::kAlias     ? "alias output"
                                                        : "concat output";
          } else if (reason == nullptr && bound.aliasOf >= 0) {
            reason = "in-place output";
          } else if (reason == nullptr && !staticHeader(bound.valueId)) {
            reason = "output without a static rank and dtype";
            noteNoHeader(bound.valueId);
          } else if (reason == nullptr && isHostWritten(bound.valueId)) {
            VLOG(1) << "Device alloc: output %" << bound.valueId << " from "
                    << producerOf_(bound.valueId) << " has a host-written "
                    << (bound.valueId < static_cast<int32_t>(
                                            state.valueShapeSlot.size()) &&
                                state.valueShapeSlot[bound.valueId] >= 0
                            ? "shape"
                            : "address");
            reason = "output with a host-written slot";
          } else if (
              reason == nullptr && isFrameRead(bound.valueId) &&
              (descriptorOffset.count(bound.valueId) == 0 ||
               state.fetchHostShapes.count(bound.valueId) != 0)) {
            reason = "graph output without a descriptor to fetch";
          }
        }
      }
    }
    // A kernel writes an in-place concat operand under the shape the setup
    // gives it; the host does not write a slot of a step on the path.
    for (const auto& concat : program.concats) {
      for (size_t i = 0; i < concat.operandIds.size(); ++i) {
        if (reason == nullptr && i < concat.inPlace.size() &&
            concat.inPlace[i] != 0 && isHostWritten(concat.operandIds[i])) {
          reason = "concat operand with a host-written slot";
        }
      }
    }
    if (reason != nullptr) {
      ++why[reason];
      offWhy[stepKey(node, step)] = reason;
    } else {
      onPath.insert(stepKey(node, step));
    }
  }
  const auto numCandidates = onPath.size();
  // A value the reader copies back stays on the path: a kernel output or a
  // view the setup writes in an export slot, or what the host makes of such
  // values.
  folly::F14FastMap<int32_t, bool> fetchable;
  std::function<bool(int32_t)> isFetchable = [&](int32_t id) -> bool {
    if (descriptorOffset.count(id) != 0) {
      return true;
    }
    auto producer = producerOf.find(id);
    if (producer != producerOf.end() && onPath.count(producer->second) == 0) {
      return true;
    }
    auto known = fetchable.find(id);
    if (known != fetchable.end()) {
      return known->second;
    }
    fetchable[id] = false;
    if (producer == producerOf.end()) {
      // Made on the host, such as a size list: known when what it is made
      // of is. A value the host computes from one the path makes would hold
      // a capacity.
      const auto& idToValue = state.waveGraph->idToValue();
      auto value = idToValue.find(id);
      const auto* node = value != idToValue.end() && value->second != nullptr
          ? value->second->producer()
          : nullptr;
      if (node != nullptr) {
        for (const auto& input : node->inputs()) {
          if (input.value != nullptr && !isFetchable(input.value->id())) {
            return false;
          }
        }
      }
      fetchable[id] = true;
      return true;
    }
    // Made on the path with no descriptor to copy back.
    return false;
  };
  for (bool changed = true; changed;) {
    changed = false;
    auto drop = [&](int64_t key, const std::string& reason) {
      if (onPath.erase(key) != 0) {
        changed = true;
        ++why[reason];
        offWhy[key] = reason;
      }
    };
    std::string readerWhy;
    auto dropProducer = [&](int32_t id, const char* reason) {
      auto it = producerOf.find(id);
      if (it != producerOf.end()) {
        drop(it->second, fmt::format("{} ({})", reason, readerWhy));
      }
    };
    fetchable.clear();
    auto readOff = [&](int32_t id, const char* reason) {
      if (!isFetchable(id)) {
        dropProducer(id, reason);
      }
    };
    for (size_t node = 0; node < state.stepVectors.size(); ++node) {
      auto planIt = planOf.find(static_cast<int32_t>(node));
      const auto* plan = planIt != planOf.end() ? planIt->second : nullptr;
      for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
        const auto& sv = state.stepVectors[node][step];
        const auto key =
            stepKey(static_cast<int32_t>(node), static_cast<int32_t>(step));
        if (sv.program != nullptr && onPath.count(key) == 0) {
          auto off = offWhy.find(key);
          readerWhy = off != offWhy.end() ? off->second.substr(0, 40)
              : !sv.allWordsFromSetup     ? "reader not fully set up"
                                          : "reader sized on the host";
          for (auto id : sv.readIds) {
            readOff(id, "read by a kernel off the path");
          }
          for (const auto& fill : sv.program->fills) {
            readOff(fill.valueId, "filled into a step off the path");
          }
          for (const auto* derives :
               {&sv.program->derives, &sv.program->viewDerives}) {
            for (const auto& derive : *derives) {
              for (auto id : derive.operands) {
                readOff(id, "derived from off the path");
              }
            }
          }
        }
        if (plan == nullptr || step >= plan->groupsByStep.size()) {
          continue;
        }
        // A group is on the path whole or not at all.
        for (auto g : plan->groupsByStep[step]) {
          std::vector<int64_t> keys = {key};
          const char* reason = nullptr;
          for (auto id : groupValues(plan->groups[g])) {
            auto it = producerOf.find(id);
            auto written = writerOf.find(id);
            if (it != producerOf.end()) {
              keys.push_back(it->second);
            } else if (written != writerOf.end()) {
              keys.push_back(written->second);
            } else {
              if (reason == nullptr) {
                VLOG(1) << "Device alloc: group value %" << id << " from "
                        << producerOf_(id) << " has no producer on a step";
              }
              reason = "group value no kernel produces";
            }
          }
          for (auto k : keys) {
            if (reason == nullptr && onPath.count(k) == 0) {
              reason = k == key ? "group carved off the path"
                                : "group value produced off the path";
            }
          }
          if (reason != nullptr) {
            for (auto k : keys) {
              drop(k, reason);
            }
          }
        }
      }
    }
  }
  state.deviceOnlyValue.assign(state.valueShapeSlot.size(), 0);
  state.staticHeaders.assign(
      state.valueShapeSlot.size(), ExecutionState::StaticHeader{});
  auto mark = [&](int32_t id) {
    if (id >= 0 && id < static_cast<int32_t>(state.deviceOnlyValue.size())) {
      state.deviceOnlyValue[id] = 1;
      if (auto header = staticHeader(id)) {
        auto& to = state.staticHeaders[id];
        to.rank = header->first;
        to.dtype = static_cast<int8_t>(header->second);
        // The dims the derived shape fixes: a record may keep the host's
        // word for those.
        const auto* shape = shapeOf(id);
        for (int32_t d = 0; d < to.rank && d < kMaxDims; ++d) {
          to.dims[d] = shape != nullptr &&
                  static_cast<size_t>(d) < shape->dims.size() &&
                  shape->dims[d].isConstant()
              ? static_cast<int32_t>(shape->dims[d].value)
              : 1;
        }
      }
    }
  };
  int32_t numOnPath = 0;
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    auto planIt = planOf.find(static_cast<int32_t>(node));
    const auto* plan = planIt != planOf.end() ? planIt->second : nullptr;
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      auto& sv = state.stepVectors[node][step];
      const auto key =
          stepKey(static_cast<int32_t>(node), static_cast<int32_t>(step));
      sv.deviceAllocStep = onPath.count(key) != 0;
      if (!sv.deviceAllocStep) {
        auto off = offWhy.find(key);
        sv.offPathReason = off != offWhy.end() ? off->second
            : !sv.allWordsFromSetup && !sv.deviceWordsReady
            ? (sv.hostWordReason.empty() ? std::string("not fully set up")
                                         : sv.hostWordReason)
            : !sv.censusReason.empty()
            ? sv.censusReason
            : std::string("not device-sized, or host reads outputs");
        if (sv.program != nullptr) {
          VLOG(1) << "Device alloc: step " << node << "." << step
                  << " off the path: " << sv.offPathReason
                  << (sv.hostWordReason.empty() ? "" : "; ")
                  << sv.hostWordReason;
        }
        continue;
      }
      ++numOnPath;
      for (const auto& launch : sv.program->launches) {
        for (const auto* outputs : {&launch.outputs, &launch.largestOutputs}) {
          for (const auto& bound : *outputs) {
            mark(bound.valueId);
          }
        }
      }
      for (const auto& derive : sv.program->derives) {
        mark(derive.output);
        for (auto id : derive.outputs) {
          mark(id);
        }
      }
      // An operand a kernel writes in place into its concat's band is an
      // output of the step that no launch binds.
      for (const auto& concat : sv.program->concats) {
        for (size_t i = 0; i < concat.operandIds.size(); ++i) {
          if (i < concat.inPlace.size() && concat.inPlace[i] != 0) {
            mark(concat.operandIds[i]);
          }
        }
      }
      if (plan != nullptr && step < plan->groupsByStep.size()) {
        for (auto g : plan->groupsByStep[step]) {
          for (auto id : groupValues(plan->groups[g])) {
            mark(id);
          }
        }
      }
    }
  }
  // The storage a view over the table views, through views, or -1.
  auto rootOf = [&](int32_t id) {
    int32_t base = -1;
    for (auto it = viewBaseOf.find(id); it != viewBaseOf.end();
         it = viewBaseOf.find(base)) {
      base = it->second;
    }
    return base;
  };
  // The path values each step off the path reads, fetched from their
  // producers' parameters.
  int32_t numFetches = 0;
  int32_t numFetchingSteps = 0;
  for (auto& steps : state.stepVectors) {
    for (auto& sv : steps) {
      sv.deviceFetches.clear();
      sv.fetchBegin = -1;
      sv.fetchEnd = -1;
    }
  }
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      auto& sv = state.stepVectors[node][step];
      if (sv.program == nullptr || sv.deviceAllocStep) {
        continue;
      }
      folly::F14FastSet<int32_t> reads(sv.readIds.begin(), sv.readIds.end());
      for (const auto& fill : sv.program->fills) {
        reads.insert(fill.valueId);
      }
      for (const auto& data : sv.standalones) {
        reads.insert(data.actualInputs.begin(), data.actualInputs.end());
      }
      for (const auto* derives :
           {&sv.program->derives, &sv.program->viewDerives}) {
        for (const auto& derive : *derives) {
          reads.insert(derive.operands.begin(), derive.operands.end());
        }
      }
      std::vector<int32_t> sorted(reads.begin(), reads.end());
      std::sort(sorted.begin(), sorted.end());
      folly::F14FastSet<int32_t> fetched;
      // Fetches 'id' from its producer's parameters if a step on the path
      // makes it.
      auto fetch = [&](int32_t id) {
        auto producer = producerOf.find(id);
        if (id < 0 || producer == producerOf.end() ||
            onPath.count(producer->second) == 0 || !fetched.insert(id).second) {
          return;
        }
        auto offsetIt = descriptorOffset.find(id);
        TORCH_CHECK(
            offsetIt != descriptorOffset.end(),
            "A fetched value has no descriptor in its producer's parameters: %",
            id);
        const auto offset = offsetIt->second;
        const auto pNode = static_cast<int32_t>(producer->second >> 32);
        const auto pStep = static_cast<int32_t>(producer->second & 0xffffffff);
        sv.deviceFetches.push_back(
            {id, pNode, pStep, offset, isFrameRead(id), rootOf(id)});
        auto& from = state.stepVectors[pNode][pStep];
        const auto end = offset + static_cast<int32_t>(sizeof(TableTensor));
        from.fetchBegin =
            from.fetchBegin < 0 ? offset : std::min(from.fetchBegin, offset);
        from.fetchEnd = std::max(from.fetchEnd, end);
        ++numFetches;
      };
      for (auto id : sorted) {
        fetch(id);
      }
      numFetchingSteps += sv.deviceFetches.empty() ? 0 : 1;
    }
  }
  // The graph outputs path steps make, fetched at the end.
  state.outputFetches.clear();
  for (const auto& [id, key] : producerOf) {
    if (onPath.count(key) == 0 || !isFrameRead(id) ||
        (static_cast<size_t>(id) < state.standaloneReadOnly.size() &&
         state.standaloneReadOnly[id] != 0)) {
      continue;
    }
    auto offsetIt = descriptorOffset.find(id);
    TORCH_CHECK(
        offsetIt != descriptorOffset.end(),
        "A graph output made on the device path has no descriptor to fetch: %",
        id);
    const auto offset = offsetIt->second;
    const auto pNode = static_cast<int32_t>(key >> 32);
    const auto pStep = static_cast<int32_t>(key & 0xffffffff);
    state.outputFetches.push_back(
        {id, pNode, pStep, offset, /*frame=*/true, rootOf(id)});
    auto& from = state.stepVectors.at(pNode).at(pStep);
    const auto end = offset + static_cast<int32_t>(sizeof(TableTensor));
    from.fetchBegin =
        from.fetchBegin < 0 ? offset : std::min(from.fetchBegin, offset);
    from.fetchEnd = std::max(from.fetchEnd, end);
  }
  LOG(INFO) << "Device alloc: " << numFetches << " values fetched by "
            << numFetchingSteps << " steps off the path, "
            << state.outputFetches.size() << " graph outputs at the end";
  LOG_IF(INFO, !noHeaderWhy.empty())
      << "Device alloc: values without a static rank and dtype:"
      << largestCounts(noHeaderWhy, 12);
  LOG(INFO) << "Device alloc: " << numOnPath << " steps on the device path, of "
            << numCandidates << " eligible and " << candidates.size()
            << " candidates; off the path:" << largestCounts(why, 12);
}

} // namespace

void DeviceSymbols::bindSizeChecks(ExecutionState& state) {
  const auto& config = WaveConfig::get();
  if (!config.deviceAlloc || config.debugSingleOps || config.launchPerOp ||
      config.referenceFrame != nullptr) {
    bindSizeChecksPass(state, nullptr, false);
    return;
  }
  // On the device path the setup places the members of device symbols'
  // regions at their capacities, which only holds for groups the host never
  // carves. The first pass places them so everywhere to pick the path; the
  // next ones only in the groups the path steps carve, until the path keeps
  // its groups. Without that the members stay where the host puts them.
  std::vector<std::pair<StepProgram*, std::shared_ptr<SymbolFixups>>> saved;
  for (auto& steps : state.stepVectors) {
    for (auto& sv : steps) {
      if (sv.program != nullptr) {
        saved.emplace_back(
            sv.program.get(),
            sv.program->symbolFixups == nullptr
                ? nullptr
                : std::make_shared<SymbolFixups>(*sv.program->symbolFixups));
      }
    }
  }
  // A pass keeps the first source of each symbol, some of them in tables
  // the next pass makes again.
  const auto savedSources = state.symbolSources;
  const auto savedSizeCheckSymbols = state.sizeCheckSymbols;
  auto restore = [&]() {
    state.symbolSources = savedSources;
    state.sizeCheckSymbols = savedSizeCheckSymbols;
    for (auto& [program, fixups] : saved) {
      program->symbolFixups =
          fixups == nullptr ? nullptr : std::make_shared<SymbolFixups>(*fixups);
      if (fixups != nullptr) {
        program->symbolFixups->deviceRecords = at::Tensor();
        program->symbolFixups->deviceCode = at::Tensor();
      }
    }
  };
  auto groupsOnPath = [&]() {
    folly::F14FastSet<const AllocGroup*> groups;
    for (const auto& node : waveGraph_.nodes()) {
      const auto* invocation = node->kernels();
      const auto* plan = invocation ? invocation->allocGroupPlan() : nullptr;
      if (plan == nullptr) {
        continue;
      }
      const auto& steps = state.stepVectors.at(invocation->sequenceNumber());
      for (size_t step = 0;
           step < steps.size() && step < plan->groupsByStep.size();
           ++step) {
        if (steps[step].deviceAllocStep) {
          for (auto g : plan->groupsByStep[step]) {
            groups.insert(&plan->groups[g]);
          }
        }
      }
    }
    return groups;
  };
  bindSizeChecksPass(state, nullptr, true);
  auto pathGroups = groupsOnPath();
  for (int32_t pass = 0; pass < 3; ++pass) {
    restore();
    bindSizeChecksPass(state, &pathGroups, false);
    auto groups = groupsOnPath();
    if (groups == pathGroups) {
      return;
    }
    pathGroups = std::move(groups);
  }
  LOG(WARNING) << "Device alloc: the path does not settle on its groups; "
                  "region members stay where the host puts them";
  restore();
  bindSizeChecksPass(state, nullptr, false);
}

void DeviceSymbols::bindSizeChecksPass(
    ExecutionState& state,
    const folly::F14FastSet<const AllocGroup*>* capacityRegion,
    bool allRegions) {
  deriveShapesFromInputs(state);
  SizeSources found(*derivedShapes_, waveGraph_.idToValue());
  placeInputSymbols(*waveGraph_.graph(), state, found);
  findStepSources(state, found);
  ElementTable table(state);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!state.sizeCheckCounts.defined()) {
    state.sizeCheckCounts = deviceLongs(10);
    state.sizeCheckTable = deviceLongs(kTableSize);
  }
  state.sizeCheckInputValues =
      deviceLongs(static_cast<int64_t>(state.sizeCheckInputs.size()));
  auto addSource = [&](int32_t index, const SizeSource& source) {
    if (state.symbolSources.size() <= static_cast<size_t>(index)) {
      state.symbolSources.resize(index + 1, {-1, -1, -1});
    }
    // A symbol device symbols picked keeps the source the kernels leave.
    if (std::get<0>(state.symbolSources[index]) != -1) {
      return;
    }
    if (source.address != 0) {
      state.symbolSources[index] = {kAddressSource, 0, source.address};
    } else if (source.node == kInputSource) {
      state.symbolSources[index] = {
          kAddressSource,
          0,
          addressOf(state.sizeCheckInputValues, source.offset)};
    } else if (source.node == kTableSource) {
      state.symbolSources[index] = {
          kAddressSource, 0, addressOf(state.sizeCheckTable, source.offset)};
    } else {
      state.symbolSources[index] = {source.node, source.step, source.offset};
    }
    state.sizeCheckSymbols.push_back(index);
  };
  auto addAddress = [&](const std::string& name, const at::Tensor& tensor) {
    const auto index = symbolIndex(name);
    addSource(index, SizeSource{kAddressSource, 0, addressOf(tensor)});
    return index;
  };
  const auto countsIndex =
      addAddress("__size_check_counts", state.sizeCheckCounts);
  const auto tableIndex =
      addAddress("__size_check_table", state.sizeCheckTable);

  // The symbols device symbols runs on capacities in this state.
  folly::F14FastSet<std::string> picked;
  for (const auto& [id, index] : state.substitutedSymbol) {
    picked.insert(symbols_[index]);
  }
  int64_t numRecords = 0;
  int64_t numUnsourced = 0;
  int64_t numCapacity = 0;
  int64_t numSkipped = 0;
  folly::F14FastMap<std::string, int64_t> unsourcedBySymbol;
  folly::F14FastMap<std::string, int64_t> noCapacityByProducer;
  // Coverage: the steps whose every tensor parameter gets its dims, strides
  // and address from the setup without reading what the host wrote, and the
  // parameters that keep a host word, by why and by producer.
  int32_t numCoverSteps = 0;
  int32_t numCovered = 0;
  int32_t numAllWords = 0;
  folly::F14FastMap<std::string, int64_t> otherByKind;
  folly::F14FastMap<std::string, int64_t> uncoveredByReason;
  folly::F14FastMap<std::string, int64_t> unplacedByProducer;
  int64_t numPlaced = 0;
  int64_t numUnplaced = 0;
  folly::F14FastMap<std::string, int64_t> memberNoSizeWhy;
  folly::F14FastMap<std::string, int64_t> regionRealWhy;
  // Whether a symbol is on the device for the setup of step 'here'.
  auto isSourcedAt = [&](Step here) {
    return [&, here](const std::string& name) {
      auto it = found.sources.find(name);
      return it != found.sources.end() &&
          (it->second.afterStep ? it->second.at() < here
                                : it->second.at() <= here) &&
          (symbolIndex_.count(name) != 0 ||
           static_cast<int32_t>(symbols_.size()) < kMaxSymbols);
    };
  };
  folly::F14FastMap<int32_t, const AllocGroupPlan*> planOf;
  for (const auto& node : waveGraph_.nodes()) {
    if (const auto* invocation = node->kernels()) {
      if (const auto* plan = invocation->allocGroupPlan()) {
        planOf[invocation->sequenceNumber()] = plan;
      }
    }
  }
  // By member of an allocation group: its data pointer, as its group's base
  // plus the aligned bytes of the members before it.
  folly::F14FastMap<int32_t, SymExpr> addressOfMember;
  // By member name: its address and element size, for the views of it.
  folly::F14FastMap<std::string, std::pair<SymExpr, int64_t>> memberByName;
  const bool writePlacement =
      WaveConfig::get().deviceSizing && WaveConfig::get().devicePlacement;
  // A concat group is laid out as its result: each operand's band starts
  // after the extents of the operands before it. The result's address comes
  // from the first band this step has a parameter for.
  // Returns the result's size in bytes, the group's demand, when the setup
  // places the group.
  auto placeConcatGroup =
      [&](Step /*here*/,
          size_t /*g*/,
          const ConcatGroupLayout& layout,
          const folly::F14FastMap<int32_t, int32_t>& /*paramOf*/,
          const std::function<bool(const std::string&)>& isSourced)
      -> std::optional<SymExpr> {
    const auto* result = found.shapeOf(layout.resultId);
    if (result == nullptr || layout.outRank <= 0 ||
        static_cast<int32_t>(result->dims.size()) != layout.outRank ||
        result->strides.size() != result->dims.size()) {
      ++numUnplaced;
      return std::nullopt;
    }
    const auto elementSize =
        static_cast<int64_t>(c10::elementSize(layout.dtype));
    const auto& stride = result->strides[layout.dim];
    std::vector<SymExpr> starts;
    SymExpr start = SymExpr::constant(0);
    for (const auto& operand : layout.operands) {
      starts.push_back(start);
      if (layout.isStack) {
        start = symAdd(start, SymExpr::constant(1));
        continue;
      }
      const auto* shape = found.shapeOf(operand.valueId);
      if (shape == nullptr ||
          static_cast<int32_t>(shape->dims.size()) != layout.outRank) {
        ++numUnplaced;
        return std::nullopt;
      }
      start = symAdd(start, shape->dims[layout.dim]);
    }
    // The bands of each operand: the carved operand and a copy's
    // destination.
    std::vector<std::pair<int32_t, size_t>> bands;
    for (size_t i = 0; i < layout.operands.size(); ++i) {
      const auto& operand = layout.operands[i];
      if (layout.memberOfOperand[i] >= 0) {
        bands.emplace_back(operand.writtenValueId(), i);
      }
      if (operand.copyDestId >= 0) {
        bands.emplace_back(operand.copyDestId, i);
      }
    }
    auto bandOffset = [&](size_t i) {
      return symMul(symMul(starts[i], stride), SymExpr::constant(elementSize));
    };
    // The result's address, which the host writes to its value's slot
    // when it carves the group.
    std::optional<SymExpr> resultAddress;
    if (layout.resultId >= 0 &&
        layout.resultId < static_cast<int32_t>(state.hasValueAddress.size())) {
      state.hasValueAddress[layout.resultId] = 1;
      resultAddress = SymExpr{
          .kind = SymExpr::Kind::kSymbol,
          .symbol = kLoadPrefix +
              std::to_string(addressOf(
                  state.addressTable,
                  state.valueAddressBase + layout.resultId))};
    }
    std::vector<std::string> names;
    if (resultAddress) {
      resultAddress->collectSymbols(names);
      for (const auto& [id, i] : bands) {
        bandOffset(i).collectSymbols(names);
      }
    }
    auto sourced = [&](const std::string& n) {
      return n.starts_with(kLoadPrefix) || isSourced(n);
    };
    if (!resultAddress || !std::all_of(names.begin(), names.end(), sourced) ||
        (writePlacement &&
         std::any_of(names.begin(), names.end(), [&](const auto& n) {
           return picked.count(n) != 0;
         }))) {
      ++numUnplaced;
      return std::nullopt;
    }
    addressOfMember[layout.resultId] = *resultAddress;
    memberByName[found.nameOf(layout.resultId)] = {*resultAddress, elementSize};
    for (const auto& [id, i] : bands) {
      addressOfMember[id] = symAdd(*resultAddress, bandOffset(i));
      memberByName[found.nameOf(id)] = {addressOfMember[id], elementSize};
    }
    SymExpr bytes = SymExpr::constant(elementSize);
    std::vector<std::string> sizeNames;
    for (const auto& dim : result->dims) {
      bytes = symMul(bytes, dim);
      dim.collectSymbols(sizeNames);
    }
    if (!std::all_of(sizeNames.begin(), sizeNames.end(), isSourced)) {
      return std::nullopt;
    }
    for (const auto& name : sizeNames) {
      if (!name.starts_with(kCapacityPrefix)) {
        addSource(symbolIndex(name), found.sources.at(name));
      }
    }
    return bytes;
  };
  // A slot per member of a group the device places, for its address.
  int64_t numMembers = 0;
  for (const auto& [seq, plan] : planOf) {
    for (const auto& group : plan->groups) {
      numMembers += static_cast<int64_t>(group.actualIds.size());
    }
  }
  state.placementTable = deviceLongs(std::max<int64_t>(1, numMembers));
  const auto& placementTable = state.placementTable;
  // A slot per group for its base, which the host writes at each carve, then
  // one per value, which the host writes at each fill of a value whose slot
  // the setup reads.
  int64_t numGroups = 0;
  for (const auto& [seq, plan] : planOf) {
    numGroups += static_cast<int64_t>(plan->groups.size());
  }
  int32_t numValues = 0;
  for (const auto& [id, value] : found.idToValue) {
    numValues = std::max(numValues, id + 1);
  }
  // Each group's base, then its capacity (the bytes of its buffer), next to
  // each other so that a carve dirties one short range; then the values.
  state.groupCapacityBase = static_cast<int32_t>(numGroups);
  state.valueAddressBase = static_cast<int32_t>(2 * numGroups);
  state.hasValueAddress.assign(numValues, 0);
  state.addressTable = deviceLongs(2 * numGroups + numValues);
  state.addressPinned = at::zeros(
      {std::max<int64_t>(1, 2 * numGroups + numValues)},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  // What each group's setup computed it needs, by group slot.
  state.demandTable = deviceLongs(std::max<int64_t>(1, numGroups));
  state.groupHostBytes.assign(numGroups, -1);
  state.groupHasDemand.assign(numGroups, 0);
  state.recordedDemand.assign(numGroups, -1);
  state.groupBaseSlot.clear();
  state.groupSlotText.clear();
  // A shape slot per value the host shapes.
  state.valueShapeSlot.assign(numValues, -1);
  int32_t numShapeSlots = 0;
  for (const auto& [id, value] : found.idToValue) {
    const auto* shape = found.shapeOf(id);
    if (isHostShaped(id) && shape != nullptr && !shape->dims.empty() &&
        shape->dims.size() <= kMaxDims) {
      state.valueShapeSlot[id] = numShapeSlots++;
      VLOG(2) << "Device symbols: shape slot for %" << id << " "
              << value->name();
    }
  }
  state.shapeTable =
      deviceLongs(std::max<int64_t>(1, 2 * kMaxDims * numShapeSlots));
  state.shapePinned = at::zeros(
      {std::max<int64_t>(1, 2 * kMaxDims * numShapeSlots)},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  // A symbol taken from a dim of a host-shaped value's parameter reads the
  // value's shape slot: the setup rewrites that parameter.
  for (auto& [name, source] : found.sources) {
    if (source.node < 0 || (source.offset & kDimsSource) == 0 ||
        static_cast<size_t>(source.node) >= state.stepVectors.size() ||
        static_cast<size_t>(source.step) >=
            state.stepVectors[source.node].size()) {
      continue;
    }
    const auto& program =
        state.stepVectors.at(source.node).at(source.step).program;
    if (program == nullptr) {
      continue;
    }
    const auto dimOffset = source.offset & ~kDimsSource;
    for (const auto& fill : program->fills) {
      const int64_t dims =
          fill.offset + static_cast<int64_t>(offsetof(Tensor, dims));
      if (fill.kind == FillOp::kTensor && fill.kernel != FillOp::kExport &&
          dimOffset >= dims &&
          dimOffset < dims + kMaxDims * static_cast<int64_t>(sizeof(int32_t)) &&
          fill.valueId < static_cast<int32_t>(state.valueShapeSlot.size()) &&
          state.valueShapeSlot[fill.valueId] >= 0) {
        source.address = addressOf(
            state.shapeTable,
            state.valueShapeSlot[fill.valueId] * 2 * kMaxDims +
                (dimOffset - dims) / static_cast<int64_t>(sizeof(int32_t)));
        break;
      }
    }
  }
  int64_t numPlacedSlots = 0;
  const auto concats = hostConcats(state);
  std::vector<bool> concatPlaced(concats.size(), false);
  // A concat the host carves as a group writes its result's address slot at
  // the carve, before any band of it is filled.
  for (const auto& [seq, plan] : planOf) {
    for (const auto& group : plan->groups) {
      if (group.concat != nullptr && group.concat->resultId >= 0 &&
          group.concat->resultId <
              static_cast<int32_t>(state.hasValueAddress.size())) {
        state.hasValueAddress[group.concat->resultId] = 1;
      }
    }
  }
  // The concat result each band is a part of.
  folly::F14FastMap<int32_t, int32_t> bandResult;
  for (const auto& concat : concats) {
    for (const auto& bands : concat.bands) {
      for (auto band : bands) {
        bandResult.emplace(band, concat.resultId);
      }
    }
  }
  // The parameters of the current step that are bands device symbols moves
  // whose address the setup writes: the result's value slot plus the band's
  // real start.
  folly::F14FastSet<int32_t> absoluteBands;
  // Why a tensor parameter got no address record, for the coverage report.
  folly::F14FastMap<int32_t, const char*> addressWhy;
  std::deque<SymShape> kernelLeft;
  // 'expr' with each symbol device symbols picks read at its capacity, which
  // is what the host lays memory out over.
  std::function<SymExpr(const SymExpr&)> atCapacity = [&](const SymExpr& expr) {
    if (expr.kind == SymExpr::Kind::kSymbol) {
      return picked.count(expr.symbol) != 0
          ? SymExpr{.kind = SymExpr::Kind::kSymbol, .symbol = kCapacityPrefix + expr.symbol}
          : expr;
    }
    auto result = expr;
    for (auto& arg : result.args) {
      arg = atCapacity(arg);
    }
    return result;
  };
  // Each value by name, for the storage roots of views. A root that is not a
  // member of a group the setup places is made on the host, by an eager op or
  // a step off the path, and the host knows its address before a step reads
  // it.
  folly::F14FastMap<std::string, int32_t> idByName;
  for (const auto& [id, value] : found.idToValue) {
    idByName.emplace(std::string(value->name()), id);
  }
  auto placeGroupMembers = [&](Step here,
                               StepProgram& program,
                               SymbolFixups& fixups) {
    program.rootAddressIds.clear();
    auto planIt = planOf.find(here.first);
    const auto* plan = planIt != planOf.end() ? planIt->second : nullptr;
    // Where a launch of the step reads each tensor; an export is only
    // written, never a source.
    folly::F14FastMap<int32_t, int32_t> paramOf;
    for (const auto& fill : program.fills) {
      if (fill.kind == FillOp::kTensor && fill.kernel != FillOp::kExport) {
        paramOf.emplace(fill.valueId, fill.offset);
      }
    }
    auto isSourced = [sourcedHere =
                          isSourcedAt(here)](const std::string& name) {
      return name.starts_with(kCapacityPrefix) || sourcedHere(name);
    };
    int32_t numTemps = 0;
    // The members of the groups carved at this step, packed in the group's
    // order at kAllocGroupAlign.
    if (plan != nullptr &&
        static_cast<size_t>(here.second) < plan->groupsByStep.size() &&
        (writePlacement || !WaveConfig::get().deviceSizing)) {
      for (auto g : plan->groupsByStep[here.second]) {
        const auto& group = plan->groups[g];
        if (group.concat != nullptr) {
          // The result's allocation is the group's arena; its demand is the
          // result's bytes.
          const auto bytes =
              placeConcatGroup(here, g, *group.concat, paramOf, isSourced);
          if (bytes && writePlacement && !program.launches.empty() &&
              numTemps < kNumTemps) {
            const auto slot =
                state.groupBaseSlot
                    .try_emplace(
                        &group,
                        static_cast<int32_t>(state.groupBaseSlot.size()))
                    .first->second;
            std::vector<int32_t> temp(kRecordWords, kHostValue);
            temp[1] = kTempRecord;
            temp[3] = numTemps;
            temp[4] = emitProgram(*bytes, fixups.code);
            fixups.records.insert(
                fixups.records.end(), temp.begin(), temp.end());
            std::vector<int32_t> record(kRecordWords, kHostValue);
            record[1] = kCapacityRecord;
            record[3] = numTemps;
            record[4] = emitProgram(
                SymExpr{
                    .kind = SymExpr::Kind::kSymbol,
                    .symbol = kLoadPrefix +
                        std::to_string(addressOf(
                            state.addressTable,
                            state.groupCapacityBase + slot))},
                fixups.code);
            const auto demand = addressOf(state.demandTable, slot);
            record[5] = static_cast<int32_t>(demand & 0xffffffff);
            record[6] =
                static_cast<int32_t>(static_cast<uint64_t>(demand) >> 32);
            fixups.records.insert(
                fixups.records.end(), record.begin(), record.end());
            fixups.numRecords += 2;
            state.groupHasDemand[slot] = 2;
            ++numTemps;
          }
          continue;
        }
        if (group.actualIds.empty() || paramOf.count(group.actualIds[0]) == 0) {
          continue;
        }
        // The aligned size of each member up to the first one without a
        // device size, which still gets its address. With every member's
        // size the scan's total is the group's demand.
        std::vector<SymExpr> sizes;
        std::string memberText;
        for (size_t i = 0; i < group.actualIds.size(); ++i) {
          const auto id = group.actualIds[i];
          const SymShape* shape = nullptr;
          if (isSetOnDevice(id)) {
            auto it = capacityByName_.find(found.nameOf(id));
            shape = it != capacityByName_.end() ? &it->second : nullptr;
          } else if (!isHostShaped(id)) {
            shape = found.shapeOf(id);
          }
          // The host sizes over the capacity of a symbol device symbols
          // picks: the frame holds the capacity in its place. An element of
          // a tensor whose memory may be reused by now is read from the
          // table, where an earlier setup copied it.
          // On the device path a region member of a group the host never
          // carves is placed at its capacity.
          const bool regionPlaced = state.symbolRegion.count(id) != 0 &&
              (allRegions ||
               (capacityRegion != nullptr &&
                capacityRegion->count(&group) != 0));
          std::vector<SymExpr> dims;
          std::vector<std::string> names;
          if (shape != nullptr) {
            for (const auto& dim : shape->dims) {
              dims.push_back(atCapacity(dim));
              dims.back().collectSymbols(names);
            }
            table.route(here, dims, names, found);
          }
          // Region census for the device path: whether the member's real
          // size (no capacities) is known to its group's setup.
          if (writePlacement && WaveConfig::get().deviceAlloc &&
              shape != nullptr && state.symbolRegion.count(id) != 0) {
            std::vector<std::string> realNames;
            for (const auto& dim : shape->dims) {
              dim.collectSymbols(realNames);
            }
            const char* why = "sourced by an earlier step";
            for (const auto& name : realNames) {
              if (isSourced(name)) {
                continue;
              }
              auto source = found.sources.find(name);
              why = source == found.sources.end() ? "no source"
                  : source->second.at() == here   ? "made in the same step"
                                                  : "made later";
              break;
            }
            ++regionRealWhy[why];
          }
          // A member device symbols sizes from a capacity is where the
          // host's capacity puts it, which its derived shape does not always
          // give. The host writes its address to its value's slot.
          if (shape == nullptr || shape->dims.size() > kMaxDims ||
              !std::all_of(names.begin(), names.end(), isSourced) ||
              (state.symbolRegion.count(id) != 0 && !regionPlaced)) {
            if (i + 1 < group.actualIds.size()) {
              ++numUnplaced;
            }
            if (writePlacement) {
              const auto* value = found.idToValue.count(id) != 0
                  ? found.idToValue.at(id)
                  : nullptr;
              const auto producer =
                  value != nullptr && value->producer() != nullptr
                  ? std::string(value->producer()->target())
                  : std::string("input");
              std::string unsourced;
              for (const auto& name : names) {
                if (!isSourced(name)) {
                  unsourced = name.substr(0, 1);
                  break;
                }
              }
              ++memberNoSizeWhy[fmt::format(
                  "{} ({})",
                  shape == nullptr
                      ? (isHostShaped(id) ? "host-shaped" : "no shape")
                      : shape->dims.size() > kMaxDims ? "rank"
                      : state.symbolRegion.count(id) != 0
                      ? "symbol region"
                      : "symbol " + unsourced + " not on the device yet",
                  producer)];
            }
            break;
          }
          for (const auto& name : names) {
            if (!name.starts_with(kCapacityPrefix)) {
              addSource(symbolIndex(name), found.sources.at(name));
            }
          }
          SymExpr bytes = SymExpr::constant(
              static_cast<int64_t>(c10::elementSize(group.dtypes[i])));
          memberText += fmt::format(" %{}:[", id);
          for (const auto& dim : dims) {
            bytes = symMul(bytes, dim);
            memberText += dim.toString() + ",";
          }
          memberText += "]";
          sizes.push_back(symMul(
              symFloorDiv(
                  symAdd(bytes, SymExpr::constant(kAllocGroupAlign - 1)),
                  SymExpr::constant(kAllocGroupAlign)),
              SymExpr::constant(kAllocGroupAlign)));
        }
        const bool complete = sizes.size() == group.actualIds.size();
        const auto numAddressed = static_cast<int32_t>(
            std::min(sizes.size() + 1, group.actualIds.size()));
        sizes.push_back(SymExpr::constant(0));
        const auto count = static_cast<int32_t>(sizes.size());
        if (numTemps + count > kNumTemps ||
            numPlacedSlots + numAddressed > placementTable.numel()) {
          ++numUnplaced;
          continue;
        }
        auto [slotIt, newSlot] = state.groupBaseSlot.try_emplace(
            &group, static_cast<int32_t>(state.groupBaseSlot.size()));
        if (newSlot) {
          getStepVectors(state.stepVectors, here.first, here.second)
              .groupBaseSlots.push_back(slotIt->second);
        }
        const auto baseName = kLoadPrefix +
            std::to_string(addressOf(state.addressTable, slotIt->second));
        auto temp = [&](int32_t i) {
          return SymExpr{
              .kind = SymExpr::Kind::kSymbol,
              .symbol = kTempPrefix + std::to_string(numTemps + i)};
        };
        auto addRecord = [&](std::vector<int32_t> record) {
          fixups.records.insert(
              fixups.records.end(), record.begin(), record.end());
          ++fixups.numRecords;
        };
        for (int32_t i = 0; i < count; ++i) {
          std::vector<int32_t> record(kRecordWords, kHostValue);
          record[1] = kTempRecord;
          record[3] = numTemps + i;
          record[4] = emitProgram(sizes[i], fixups.code);
          addRecord(std::move(record));
        }
        {
          std::vector<int32_t> record(kRecordWords, kHostValue);
          record[1] = kScanRecord;
          record[3] = numTemps;
          record[4] = count;
          addRecord(std::move(record));
        }
        // A step that launches nothing runs no setup.
        if (writePlacement && complete && !program.launches.empty()) {
          std::vector<int32_t> record(kRecordWords, kHostValue);
          record[1] = kCapacityRecord;
          record[3] = numTemps + count - 1;
          record[4] = emitProgram(
              SymExpr{
                  .kind = SymExpr::Kind::kSymbol,
                  .symbol = kLoadPrefix +
                      std::to_string(addressOf(
                          state.addressTable,
                          state.groupCapacityBase + slotIt->second))},
              fixups.code);
          state.groupHasDemand[slotIt->second] = 1;
          if (state.groupSlotText.size() <=
              static_cast<size_t>(slotIt->second)) {
            state.groupSlotText.resize(slotIt->second + 1);
          }
          state.groupSlotText[slotIt->second] = memberText;
          const auto demand = addressOf(state.demandTable, slotIt->second);
          record[5] = static_cast<int32_t>(demand & 0xffffffff);
          record[6] = static_cast<int32_t>(static_cast<uint64_t>(demand) >> 32);
          addRecord(std::move(record));
        }
        const SymExpr base{.kind = SymExpr::Kind::kSymbol, .symbol = baseName};
        for (int32_t i = 0; i < numAddressed; ++i) {
          const auto id = group.actualIds[i];
          const auto dest = addressOf(placementTable, numPlacedSlots + i);
          std::vector<int32_t> record(kRecordWords, kHostValue);
          record[1] = kStoreRecord;
          record[4] = emitProgram(
              symAdd(
                  base,
                  symAdd(temp(i), symMul(SymExpr::constant(-1), temp(0)))),
              fixups.code);
          record[5] = static_cast<int32_t>(dest & 0xffffffff);
          record[6] = static_cast<int32_t>(static_cast<uint64_t>(dest) >> 32);
          addRecord(std::move(record));
          addressOfMember[id] = SymExpr{
              .kind = SymExpr::Kind::kSymbol,
              .symbol = kLoadPrefix + std::to_string(dest)};
          memberByName[found.nameOf(id)] = {
              addressOfMember[id],
              static_cast<int64_t>(c10::elementSize(group.dtypes[i]))};
        }
        numTemps += count;
        numPlacedSlots += numAddressed;
      }
    }
    // The bands of fused concats whose result is placed: an operand's band
    // starts after the extents of the operands before it.
    for (size_t c = 0; c < concats.size(); ++c) {
      const auto& concat = concats[c];
      auto result = memberByName.find(found.nameOf(concat.resultId));
      const auto* shape = found.shapeOf(concat.resultId);
      if (concatPlaced[c] || result == memberByName.end() || shape == nullptr ||
          concat.dim < 0 ||
          static_cast<size_t>(concat.dim) >= shape->strides.size()) {
        continue;
      }
      concatPlaced[c] = true;
      SymExpr start = SymExpr::constant(0);
      for (size_t i = 0; i < concat.operands.size(); ++i) {
        const auto* operand = found.shapeOf(concat.operands[i]);
        if (!concat.stack &&
            (operand == nullptr ||
             operand->dims.size() != shape->dims.size())) {
          break;
        }
        const auto address = symAdd(
            result->second.first,
            symMul(
                symMul(start, shape->strides[concat.dim]),
                SymExpr::constant(result->second.second)));
        for (auto band : concat.bands[i]) {
          if (addressOfMember.emplace(band, address).second) {
            memberByName[found.nameOf(band)] = {address, result->second.second};
          }
        }
        if (concat.stack) {
          start = symAdd(start, SymExpr::constant(1));
        } else {
          start = symAdd(start, operand->dims[concat.dim]);
        }
      }
    }
    // Every parameter of a placed member or of a view of one, in its
    // group's step and later. Device symbols moves the starts of some views
    // and bands itself.
    for (const auto& fill : program.fills) {
      if (fill.kind != FillOp::kTensor) {
        continue;
      }
      const auto id = fill.valueId;
      const auto param = fill.offset;
      std::optional<SymExpr> placed;
      // An offset in elements of the parameter's size, for a view of a graph
      // input.
      std::optional<SymExpr> elements;
      bool valueSlot = false;
      int32_t rootAddress = -1;
      auto it = addressOfMember.find(id);
      const auto* view = found.shapeOf(id);
      auto input = view != nullptr && !view->base.empty()
          ? found.inputAddress.find(view->base)
          : found.inputAddress.find(found.nameOf(id));
      if (it != addressOfMember.end()) {
        placed = it->second;
      } else if (
          view != nullptr && !view->base.empty() &&
          memberByName.count(view->base) != 0) {
        const auto& member = memberByName.at(view->base);
        placed = symAdd(
            member.first,
            symMul(view->offset, SymExpr::constant(member.second)));
      } else if (input != found.inputAddress.end()) {
        placed = SymExpr{
            .kind = SymExpr::Kind::kSymbol,
            .symbol = kLoadPrefix +
                std::to_string(addressOf(
                    state.sizeCheckInputValues, input->second))};
        if (view != nullptr && !view->base.empty()) {
          elements = view->offset;
        }
      } else if (
          view != nullptr && !view->base.empty() &&
          idByName.count(view->base) != 0) {
        // A view of a value the host made: the host writes the value's
        // address to its slot before the step, so the view needs no address
        // of its own.
        const auto root = idByName.at(view->base);
        placed = SymExpr{
            .kind = SymExpr::Kind::kSymbol,
            .symbol = kLoadPrefix +
                std::to_string(addressOf(
                    state.addressTable, state.valueAddressBase + root))};
        elements = view->offset;
        rootAddress = root;
      } else if (
          id >= 0 && id < static_cast<int32_t>(state.hasValueAddress.size()) &&
          fill.kernel != FillOp::kExport) {
        // Memory the host chose some other way: the address of the value's
        // last fill. No fill writes an export's.
        placed = SymExpr{
            .kind = SymExpr::Kind::kSymbol,
            .symbol = kLoadPrefix +
                std::to_string(addressOf(
                    state.addressTable, state.valueAddressBase + id))};
        valueSlot = true;
      }
      const bool absoluteBand = absoluteBands.count(param) != 0;
      if (absoluteBand) {
        placed = SymExpr{
            .kind = SymExpr::Kind::kSymbol,
            .symbol = kLoadPrefix +
                std::to_string(addressOf(
                    state.addressTable,
                    state.valueAddressBase + bandResult.at(id)))};
        elements = state.bandStarts.at(id);
        valueSlot = false;
        rootAddress = -1;
      } else if (
          !placed || state.viewStarts.count(id) != 0 ||
          state.bandStarts.count(id) != 0) {
        addressWhy[param] = !placed           ? "address: none"
            : state.viewStarts.count(id) != 0 ? "address: view start"
                                              : "address: band start";
        continue;
      }
      std::vector<SymExpr> address{*placed};
      if (elements) {
        address.push_back(*elements);
      }
      // A band's start is its real one: the kernels write the concat with
      // the real extents.
      if (!absoluteBand) {
        for (auto& part : address) {
          part = atCapacity(part);
        }
      }
      std::vector<std::string> names;
      for (const auto& part : address) {
        part.collectSymbols(names);
      }
      names.erase(
          std::remove_if(
              names.begin(),
              names.end(),
              [&](const auto& n) {
                // Device symbols sources the symbols it picks.
                return n.starts_with(kLoadPrefix) ||
                    n.starts_with(kCapacityPrefix) || picked.count(n) != 0;
              }),
          names.end());
      // The base member's own parameter is the base.
      if (address.size() == 1 && address[0].kind == SymExpr::Kind::kSymbol &&
          found.sources.count(address[0].symbol) != 0 &&
          found.sources.at(address[0].symbol).at() == here) {
        addressWhy[param] = "address: own base";
        continue;
      }
      table.route(here, address, names, found);
      if (!std::all_of(names.begin(), names.end(), isSourced)) {
        addressWhy[param] = "address: unsourced";
        ++numUnplaced;
        continue;
      }
      for (const auto& name : names) {
        addSource(symbolIndex(name), found.sources.at(name));
      }
      if (valueSlot) {
        state.hasValueAddress[id] = 1;
      }
      if (rootAddress >= 0) {
        VLOG(1) << "Device alloc: %" << id << " at step " << here.first << "."
                << here.second << " over host-made %" << rootAddress;
        program.rootAddressIds.push_back(rootAddress);
      }
      std::vector<int32_t> record(kRecordWords, kHostValue);
      record[0] = param;
      record[1] = writePlacement ? kPlacedRecord : kPlacementRecord;
      record[3] = countsIndex | (id << 10);
      record[4] = emitProgram(address[0], fixups.code);
      if (address.size() > 1) {
        record[5] = emitProgram(address[1], fixups.code);
      }
      fixups.records.insert(fixups.records.end(), record.begin(), record.end());
      ++fixups.numRecords;
      ++numPlaced;
    }
  };
  for (size_t node = 0; node < state.stepVectors.size(); ++node) {
    for (size_t step = 0; step < state.stepVectors[node].size(); ++step) {
      auto& program = state.stepVectors[node][step].program;
      if (program == nullptr) {
        continue;
      }
      const Step here{static_cast<int32_t>(node), static_cast<int32_t>(step)};
      folly::F14FastSet<int32_t> outputs;
      for (const auto& launch : program->launches) {
        for (const auto& bound : launch.outputs) {
          outputs.insert(bound.valueId);
        }
      }
      auto fixups = program->symbolFixups != nullptr
          ? program->symbolFixups
          : std::make_shared<SymbolFixups>();
      folly::F14FastSet<int32_t> sized;
      absoluteBands.clear();
      // The device symbols record of each parameter, by offset; the records
      // before 'numSymbolRecords' are device symbols'.
      folly::F14FastMap<int32_t, int32_t> symbolRecordOf;
      const int32_t numSymbolRecords = fixups->numRecords;
      for (int32_t r = 0; r < fixups->numRecords; ++r) {
        const auto offset = fixups->records.at(r * kRecordWords);
        sized.insert(offset);
        const auto kind = fixups->records.at(r * kRecordWords + 1);
        if (kind == kTensorRecord || kind == kShapeRecord ||
            kind == kBandRecord || kind == kViewRecord) {
          symbolRecordOf[offset] = r;
        }
      }
      // The parameters whose device symbols record a full record replaces.
      folly::F14FastSet<int32_t> replaced;
      // Why a tensor parameter got no size record, for the coverage report.
      folly::F14FastMap<int32_t, const char*> unsizedWhy;
      // The dims a kernel wrote to its output parameter, as symbols sourced
      // there, and dense strides over them, for a value it sized that has
      // no symbolic shape. The rank is its capacity's.
      auto kernelLeftShape = [&](int32_t id, Step at) -> const SymShape* {
        const auto name = found.nameOf(id);
        auto writers = found.producedAt.find(name);
        auto capacity = capacityByName_.find(name);
        if (writers == found.producedAt.end() ||
            capacity == capacityByName_.end() ||
            capacity->second.dims.empty() ||
            capacity->second.dims.size() > kMaxDims) {
          return nullptr;
        }
        const SizeSource* writer = nullptr;
        for (const auto& candidate : writers->second) {
          if (candidate.at() < at) {
            writer = &candidate;
          }
        }
        if (writer == nullptr) {
          return nullptr;
        }
        SymShape shape;
        const auto rank = capacity->second.dims.size();
        for (size_t d = 0; d < rank; ++d) {
          const auto symbol = fmt::format(
              "u_kernel_{}_{}_{}_{}", name, writer->node, writer->step, d);
          found.sources.try_emplace(
              symbol,
              SizeSource{
                  writer->node,
                  writer->step,
                  dimSource(static_cast<int32_t>(writer->offset), d),
                  true});
          shape.dims.push_back(
              SymExpr{.kind = SymExpr::Kind::kSymbol, .symbol = symbol});
        }
        shape.strides.resize(rank);
        SymExpr stride = SymExpr::constant(1);
        for (size_t d = rank; d-- > 0;) {
          shape.strides[d] = stride;
          stride = symMul(stride, shape.dims[d]);
        }
        return &kernelLeft.emplace_back(std::move(shape));
      };
      folly::F14FastMap<int32_t, const char*> strideWhy;
      // The words other than tensor parameters the setup writes: the ones
      // the kernels change, and integer scalars with a symbolic form.
      folly::F14FastSet<int32_t> otherWritten;
      for (const auto& fill : program->fills) {
        if (!WaveConfig::get().deviceSizing || !writePlacement) {
          break;
        }
        if (fill.kind == FillOp::kZero64 || fill.kind == FillOp::kZero32 ||
            fill.kind == FillOp::kUninited) {
          std::vector<int32_t> record(kRecordWords, kHostValue);
          record[0] = fill.offset;
          record[1] = kVolatileRecord;
          record[2] = fill.kind == FillOp::kZero64 ? 0
              : fill.kind == FillOp::kZero32       ? 1
                                                   : 2;
          fixups->records.insert(
              fixups->records.end(), record.begin(), record.end());
          ++fixups->numRecords;
          otherWritten.insert(fill.offset);
          continue;
        }
        if (fill.kind != FillOp::kScalar || sized.count(fill.offset) != 0) {
          continue;
        }
        auto scalar =
            found.shapes.scalarByName.find(found.nameOf(fill.valueId));
        if (scalar == found.shapes.scalarByName.end()) {
          continue;
        }
        // The host computes it over the capacities of the symbols device
        // symbols picks. A symbol read from this step's own parameters is a
        // word the host writes.
        std::vector<SymExpr> value{atCapacity(scalar->second)};
        std::vector<std::string> names;
        value[0].collectSymbols(names);
        table.route(here, value, names, found);
        auto sourcedHere = isSourcedAt(here);
        const bool sourced =
            std::all_of(names.begin(), names.end(), [&](const auto& name) {
              if (name.starts_with(kCapacityPrefix)) {
                return true;
              }
              if (!sourcedHere(name)) {
                return false;
              }
              const auto& source = found.sources.at(name);
              return source.address != 0 || source.at() != here;
            });
        if (!sourced) {
          continue;
        }
        for (const auto& name : names) {
          if (!name.starts_with(kCapacityPrefix)) {
            addSource(symbolIndex(name), found.sources.at(name));
          }
        }
        std::vector<int32_t> record(kRecordWords, kHostValue);
        record[0] = fill.offset;
        record[1] = kScalarRecord;
        record[4] = emitProgram(value[0], fixups->code);
        fixups->records.insert(
            fixups->records.end(), record.begin(), record.end());
        ++fixups->numRecords;
        otherWritten.insert(fill.offset);
      }
      for (const auto& fill : program->fills) {
        // An export holds nothing the host wrote to check, so only a setup
        // that writes parameters fills it.
        if ((fill.kind != FillOp::kTensor && fill.kind != FillOp::kShapeOnly) ||
            (fill.kernel == FillOp::kExport &&
             !(WaveConfig::get().deviceSizing && writePlacement))) {
          continue;
        }
        // An output this step's kernel shapes holds a capacity until the
        // kernel runs: the check compares the reserve's symbolic form.
        const bool capacity = isSetOnDevice(fill.valueId) &&
            outputs.count(fill.valueId) != 0 && !WaveConfig::get().deviceSizing;
        const SymShape* shape = nullptr;
        if (capacity) {
          auto it = capacityByName_.find(found.nameOf(fill.valueId));
          if (it == capacityByName_.end()) {
            const auto* value = found.idToValue.at(fill.valueId);
            ++noCapacityByProducer
                [value->producer() != nullptr
                     ? std::string(value->producer()->target())
                     : "input"];
          } else {
            shape = &it->second;
            ++numCapacity;
          }
        } else if (
            !isSetOnDevice(fill.valueId) || outputs.count(fill.valueId) == 0) {
          // A value a kernel sized is read with the dims that kernel left.
          shape = found.shapeOf(fill.valueId);
          if (shape == nullptr && isSetOnDevice(fill.valueId)) {
            shape = kernelLeftShape(fill.valueId, here);
          }
        } else if (WaveConfig::get().deviceSizing) {
          // The kernel's own output holds the reserve's capacity until the
          // kernel writes its real dims.
          auto it = capacityByName_.find(found.nameOf(fill.valueId));
          shape = it != capacityByName_.end() ? &it->second : nullptr;
        }
        // A value the host shapes gets its dims and strides from its shape
        // slot, which the host writes at each fill.
        SymShape slotShape;
        if (shape != nullptr && isHostShaped(fill.valueId) &&
            WaveConfig::get().deviceSizing &&
            fill.valueId < static_cast<int32_t>(state.valueShapeSlot.size()) &&
            state.valueShapeSlot[fill.valueId] >= 0) {
          const auto first = state.valueShapeSlot[fill.valueId] * 2 * kMaxDims;
          auto load = [&](int32_t word) {
            return SymExpr{
                .kind = SymExpr::Kind::kSymbol,
                .symbol = kLoadPrefix +
                    std::to_string(addressOf(state.shapeTable, first + word))};
          };
          for (size_t d = 0; d < shape->dims.size(); ++d) {
            slotShape.dims.push_back(load(static_cast<int32_t>(d)));
            slotShape.strides.push_back(
                load(static_cast<int32_t>(kMaxDims + d)));
          }
          shape = &slotShape;
        } else if (shape != nullptr && isHostShaped(fill.valueId)) {
          shape = nullptr;
        }
        if (shape == nullptr || shape->dims.empty() ||
            shape->dims.size() > kMaxDims) {
          unsizedWhy[fill.offset] = isHostShaped(fill.valueId) ? "host shaped"
              : shape == nullptr && isSetOnDevice(fill.valueId)
              ? (outputs.count(fill.valueId) != 0
                     ? "kernel-sized output"
                     : "read of a kernel-sized value")
              : shape == nullptr    ? "no derived shape"
              : shape->dims.empty() ? nullptr
                                    : "rank";
          ++numSkipped;
          continue;
        }
        // A value outside device symbols' regions is sized by the host over
        // the capacities of the symbols device symbols picks: the frame holds
        // them in the symbols' place.
        const bool inRegion = state.symbolRegion.count(fill.valueId) != 0;
        std::vector<SymExpr> dims = shape->dims;
        // An export carries the real dims, which the host copies back; its
        // strides are the layout, over the capacities like the rest.
        if (!inRegion && fill.kernel != FillOp::kExport) {
          for (auto& dim : dims) {
            dim = atCapacity(dim);
          }
        }
        std::vector<std::string> names;
        for (auto& dim : dims) {
          dim.collectSymbols(names);
        }
        const bool dataSymbols =
            std::any_of(names.begin(), names.end(), isDataSymbol);
        auto isSourced = [sourcedHere =
                              isSourcedAt(here)](const std::string& name) {
          return name.starts_with(kCapacityPrefix) ||
              name.starts_with(kLoadPrefix) || sourcedHere(name);
        };
        table.route(here, dims, names, found);
        const bool sourced = std::all_of(names.begin(), names.end(), isSourced);
        if (!sourced) {
          unsizedWhy[fill.offset] = "unsourced";
          ++numUnsourced;
          for (const auto& name : names) {
            if (found.sources.count(name) != 0) {
              ++unsourcedBySymbol[name + "(later)"];
            } else {
              ++unsourcedBySymbol[name];
            }
          }
          continue;
        }
        for (const auto& name : names) {
          if (!name.starts_with(kCapacityPrefix) &&
              !name.starts_with(kLoadPrefix)) {
            addSource(symbolIndex(name), found.sources.at(name));
          }
        }
        // A stride is checked when its symbols are on the device too.
        std::vector<SymExpr> strides;
        if (fill.kind == FillOp::kTensor && !capacity &&
            shape->strides.size() == shape->dims.size()) {
          strides = shape->strides;
          if (!inRegion) {
            for (auto& stride : strides) {
              stride = atCapacity(stride);
            }
          }
          std::vector<std::string> strideNames;
          for (auto& stride : strides) {
            stride.collectSymbols(strideNames);
          }
          table.route(here, strides, strideNames, found);
        }
        std::vector<int32_t> record(kRecordWords, kHostValue);
        record[0] = fill.offset;
        if (WaveConfig::get().deviceSizing) {
          // A parameter device symbols already rewrites keeps that record. A
          // value the host sizes from a capacity, and a size over a symbol
          // device symbols picks, are left to those records, which empty the
          // tensor when a real value is above its capacity.
          const bool region = inRegion;
          if (region || !sized.insert(fill.offset).second) {
            // A value device symbols sizes gets every dim and stride from
            // the setup, in a record of device symbols' kind, which empties
            // the tensor when a real value is above its capacity. A band or
            // view keeps device symbols' start and strides: the host knows
            // the layout of a band.
            auto symbolRecord = symbolRecordOf.find(fill.offset);
            const int32_t* previous = symbolRecord != symbolRecordOf.end()
                ? fixups->records.data() + symbolRecord->second * kRecordWords
                : nullptr;
            const bool band = state.bandStarts.count(fill.valueId) != 0 ||
                state.viewStarts.count(fill.valueId) != 0;
            // A band of a concat whose result has an address slot gets its
            // address from the slot and its real start, with no move of the
            // host's address, and the result's strides at the capacities,
            // which is the host's layout of the result.
            std::vector<SymExpr> bandStrides;
            if (fill.kind == FillOp::kTensor &&
                state.bandStarts.count(fill.valueId) != 0 &&
                state.viewStarts.count(fill.valueId) == 0) {
              auto result = bandResult.find(fill.valueId);
              const auto* resultShape = result != bandResult.end()
                  ? found.shapeOf(result->second)
                  : nullptr;
              std::vector<std::string> bandNames;
              state.bandStarts.at(fill.valueId).collectSymbols(bandNames);
              if (resultShape != nullptr &&
                  resultShape->strides.size() == dims.size() &&
                  result->second >= 0 &&
                  result->second <
                      static_cast<int32_t>(state.hasValueAddress.size()) &&
                  state.hasValueAddress[result->second] != 0) {
                for (const auto& stride : resultShape->strides) {
                  bandStrides.push_back(atCapacity(stride));
                  bandStrides.back().collectSymbols(bandNames);
                }
                table.route(here, bandStrides, bandNames, found);
                if (!std::all_of(
                        bandNames.begin(),
                        bandNames.end(),
                        [&](const auto& name) {
                          return picked.count(name) != 0 || isSourced(name);
                        })) {
                  bandStrides.clear();
                }
              }
            }
            const bool absolute = !bandStrides.empty();
            bool full = (region || absolute) && strides.size() == dims.size() &&
                (fill.kind == FillOp::kShapeOnly || !strides.empty());
            for (size_t d = 0; full && !band && d < strides.size(); ++d) {
              std::vector<std::string> strideNames;
              strides.at(d).collectSymbols(strideNames);
              full = std::all_of(
                  strideNames.begin(), strideNames.end(), isSourced);
            }
            if (!full) {
              unsizedWhy[fill.offset] = region ? "device symbols: region"
                                               : "device symbols: has record";
              continue;
            }
            record[1] = fill.kind == FillOp::kShapeOnly ? kShapeRecord
                : previous != nullptr && previous[1] == kViewRecord
                ? kViewRecord
                : kTensorRecord;
            record[2] = static_cast<int32_t>(dims.size());
            record[3] =
                previous != nullptr && !absolute ? previous[3] : kHostValue;
            if (absolute) {
              absoluteBands.insert(fill.offset);
              for (auto& stride : bandStrides) {
                std::vector<std::string> strideNames;
                stride.collectSymbols(strideNames);
                for (const auto& name : strideNames) {
                  if (!name.starts_with(kCapacityPrefix) &&
                      !name.starts_with(kLoadPrefix)) {
                    addSource(symbolIndex(name), found.sources.at(name));
                  }
                }
              }
            }
            for (size_t d = 0; d < dims.size(); ++d) {
              record[4 + d] = emitProgram(dims[d], fixups->code);
              record[8 + d] = absolute
                  ? emitProgram(bandStrides[d], fixups->code)
                  : band || fill.kind == FillOp::kShapeOnly
                  ? (previous != nullptr && previous[1] != kBandRecord
                         ? previous[8 + d]
                         : kHostValue)
                  : emitProgram(strides[d], fixups->code);
            }
            for (const auto& name : names) {
              if (!name.starts_with(kCapacityPrefix) &&
                  !name.starts_with(kLoadPrefix)) {
                addSource(symbolIndex(name), found.sources.at(name));
              }
            }
            if (!band && fill.kind == FillOp::kTensor) {
              for (auto& stride : strides) {
                std::vector<std::string> strideNames;
                stride.collectSymbols(strideNames);
                for (const auto& name : strideNames) {
                  if (!name.starts_with(kCapacityPrefix) &&
                      !name.starts_with(kLoadPrefix)) {
                    addSource(symbolIndex(name), found.sources.at(name));
                  }
                }
              }
            }
            replaced.insert(fill.offset);
            fixups->records.insert(
                fixups->records.end(), record.begin(), record.end());
            ++fixups->numRecords;
            ++numRecords;
            continue;
          }
          record[1] = fill.kind == FillOp::kShapeOnly ? kSizedShapeRecord
                                                      : kSizedTensorRecord;
          record[2] = static_cast<int32_t>(dims.size());
        } else {
          record[1] = kCheckRecord;
          record[2] = static_cast<int32_t>(dims.size()) |
              ((dataSymbols ? kCheckDataSymbols : 0) |
               (fill.kind == FillOp::kShapeOnly ? kCheckShapeOnly : 0) |
               (capacity ? kCheckCapacity : 0))
                  << 8;
          record[3] = countsIndex | (fill.valueId << 10);
        }
        for (size_t d = 0; d < dims.size(); ++d) {
          record[4 + d] = emitProgram(dims[d], fixups->code);
        }
        if (fill.kind == FillOp::kTensor && strides.empty()) {
          strideWhy[fill.offset] =
              capacity ? "stride: capacity" : "stride: none derived";
        }
        for (size_t d = 0; d < strides.size(); ++d) {
          std::vector<std::string> strideNames;
          strides[d].collectSymbols(strideNames);
          if (!std::all_of(strideNames.begin(), strideNames.end(), isSourced)) {
            strideWhy[fill.offset] = "stride: unsourced";
            continue;
          }
          for (const auto& name : strideNames) {
            if (!name.starts_with(kCapacityPrefix) &&
                !name.starts_with(kLoadPrefix)) {
              addSource(symbolIndex(name), found.sources.at(name));
            }
          }
          record[8 + d] = emitProgram(strides[d], fixups->code);
        }
        fixups->records.insert(
            fixups->records.end(), record.begin(), record.end());
        ++fixups->numRecords;
        ++numRecords;
      }
      // Drop the device symbols records the full records replace.
      if (!replaced.empty()) {
        std::vector<int32_t> kept;
        int32_t numKept = 0;
        for (int32_t r = 0; r < fixups->numRecords; ++r) {
          const auto* rec = fixups->records.data() + r * kRecordWords;
          if (r < numSymbolRecords && replaced.count(rec[0]) != 0 &&
              (rec[1] == kTensorRecord || rec[1] == kShapeRecord ||
               rec[1] == kBandRecord || rec[1] == kViewRecord)) {
            continue;
          }
          kept.insert(kept.end(), rec, rec + kRecordWords);
          ++numKept;
        }
        fixups->records = std::move(kept);
        fixups->numRecords = numKept;
      }
      addressWhy.clear();
      placeGroupMembers(here, *program, *fixups);
      {
        // Per parameter offset: dims and strides all written, an address
        // written, a record that adjusts a host word.
        folly::F14FastSet<int32_t> withShape;
        folly::F14FastMap<int32_t, const char*> partShape;
        folly::F14FastSet<int32_t> withAddress;
        folly::F14FastSet<int32_t> relative;
        for (int32_t r = 0; r < fixups->numRecords; ++r) {
          const auto* rec = fixups->records.data() + r * kRecordWords;
          const auto kind = rec[1];
          if (kind == kPlacedRecord || kind == kPlacementRecord) {
            withAddress.insert(rec[0]);
          } else if (
              kind == kSizedTensorRecord || kind == kSizedShapeRecord ||
              kind == kCheckRecord ||
              (replaced.count(rec[0]) != 0 &&
               (kind == kTensorRecord || kind == kShapeRecord ||
                kind == kViewRecord))) {
            if (kind == kViewRecord || rec[3] != kHostValue) {
              relative.insert(rec[0]);
            }
            const auto rank = rec[2] & 0xff;
            // A shape-only parameter has no strides to write.
            const bool shapeOnly = kind == kSizedShapeRecord ||
                kind == kShapeRecord ||
                (kind == kCheckRecord && ((rec[2] >> 8) & kCheckShapeOnly));
            const char* missing = nullptr;
            for (int32_t d = 0; d < rank; ++d) {
              missing = rec[4 + d] == kHostValue ? "host dim"
                  : !shapeOnly && rec[8 + d] == kHostValue && missing == nullptr
                  ? "host stride"
                  : missing;
            }
            if (missing == nullptr) {
              withShape.insert(rec[0]);
            } else {
              partShape[rec[0]] = missing;
            }
          } else if (
              kind == kBandRecord || kind == kViewRecord ||
              ((kind == kTensorRecord || kind == kShapeRecord) &&
               rec[3] != kHostValue)) {
            relative.insert(rec[0]);
          }
        }
        bool covered = !program->launches.empty();
        program->placedExports.clear();
        std::string hostWord;
        auto producerName = [&](int32_t id) -> std::string {
          auto it = found.idToValue.find(id);
          const auto* producer =
              it == found.idToValue.end() ? nullptr : it->second->producer();
          return producer != nullptr ? std::string(producer->target())
                                     : "input";
        };
        for (const auto& fill : program->fills) {
          if (fill.kind != FillOp::kTensor && fill.kind != FillOp::kShapeOnly) {
            continue;
          }
          // A 0-d tensor has no dims to write.
          auto unsized = unsizedWhy.find(fill.offset);
          if (unsized != unsizedWhy.end() && unsized->second == nullptr) {
            withShape.insert(fill.offset);
          }
          const char* reason = relative.count(fill.offset) != 0 ? "relative"
              : partShape.count(fill.offset) != 0
              ? (partShape.at(fill.offset)[5] == 's' &&
                         strideWhy.count(fill.offset) != 0
                     ? strideWhy.at(fill.offset)
                     : partShape.at(fill.offset))
              : withShape.count(fill.offset) == 0
              ? (unsizedWhy.count(fill.offset) != 0 ? unsizedWhy.at(fill.offset)
                                                    : "no size record")
              : fill.kind == FillOp::kTensor &&
                  withAddress.count(fill.offset) == 0
              ? (addressWhy.count(fill.offset) != 0 ? addressWhy.at(fill.offset)
                                                    : "address")
              : nullptr;
          // An export no launch reads: fetchable if placed, else not, and
          // either way not a word of a launch.
          if (fill.kernel == FillOp::kExport) {
            if (reason == nullptr) {
              program->placedExports.insert(fill.offset);
            }
            continue;
          }
          if (reason == nullptr) {
            continue;
          }
          covered = false;
          ++uncoveredByReason[reason];
          if (hostWord.empty()) {
            hostWord = fmt::format(
                "host word: {} of %{} from {}",
                reason,
                fill.valueId,
                producerName(fill.valueId));
          }
          if ((reason[0] == 'a' && reason[1] == 'd') || reason[0] == 'r' ||
              reason[0] == 'h' || reason[0] == 'n') {
            const auto* value = found.idToValue.at(fill.valueId);
            const auto* shape = found.shapeOf(fill.valueId);
            ++unplacedByProducer
                [shape != nullptr && !shape->base.empty() ? "view"
                     : value->producer() != nullptr
                     ? std::string(value->producer()->target())
                     : "input"];
          }
        }
        numCoverSteps += program->launches.empty() ? 0 : 1;
        numCovered += covered ? 1 : 0;
        // Every word of the block: also the scalars and the words the
        // kernels change. Device symbols' scalar records count.
        bool allWords = covered;
        // Or every word but scalars the host has at the start: graph inputs
        // and constants, which the fills of a step on the device path write.
        bool inputScalarsOnly = covered;
        for (const auto& fill : program->fills) {
          if (!allWords && !inputScalarsOnly) {
            break;
          }
          if (fill.kind == FillOp::kTensor || fill.kind == FillOp::kShapeOnly) {
            continue;
          }
          if (otherWritten.count(fill.offset) != 0) {
            continue;
          }
          bool symbolRecord = false;
          for (int32_t r = 0; r < fixups->numRecords && !symbolRecord; ++r) {
            const auto* rec = fixups->records.data() + r * kRecordWords;
            symbolRecord = rec[0] == fill.offset && rec[1] == kScalarRecord;
          }
          if (!symbolRecord) {
            allWords = false;
            inputScalarsOnly = inputScalarsOnly &&
                fill.kind == FillOp::kScalar &&
                producerName(fill.valueId) == "input";
            hostWord = fmt::format(
                "host word: {} %{} from {}",
                fill.kind == FillOp::kScalar ? "scalar" : "other fill",
                fill.valueId,
                producerName(fill.valueId));
            ++otherByKind
                [fill.kind == FillOp::kScalar ? "scalar" : "other fill"];
          }
        }
        numAllWords += allWords ? 1 : 0;
        auto& coveredSv =
            getStepVectors(state.stepVectors, here.first, here.second);
        coveredSv.allWordsFromSetup =
            allWords && WaveConfig::get().deviceSizing && writePlacement;
        coveredSv.hostWordReason = std::move(hostWord);
        coveredSv.deviceWordsReady = (allWords || inputScalarsOnly) &&
            WaveConfig::get().deviceSizing && writePlacement;
      }
      if (fixups->numRecords > 0) {
        program->symbolFixups = std::move(fixups);
      }
    }
  }
  for (const auto& [at, names] : table.copies) {
    auto& program = state.stepVectors.at(at.first).at(at.second).program;
    if (program->symbolFixups == nullptr) {
      program->symbolFixups = std::make_shared<SymbolFixups>();
    }
    auto& fixups = *program->symbolFixups;
    for (const auto& name : names) {
      const auto& source = found.sources.at(name);
      if (source.node == std::numeric_limits<int32_t>::max()) {
        continue;
      }
      const auto index = symbolIndex(name);
      addSource(index, source);
      std::vector<int32_t> record(kRecordWords, 0);
      record[1] = kCopyRecord;
      record[3] = index;
      record[4] = table.slots.at(name);
      record[5] = tableIndex;
      fixups.records.insert(fixups.records.end(), record.begin(), record.end());
      ++fixups.numRecords;
    }
  }
  if (state.symbolCapacities.size() < state.symbolSources.size()) {
    state.symbolCapacities.resize(state.symbolSources.size(), 0);
  }
  for (auto index : state.sizeCheckSymbols) {
    state.symbolCapacities.at(index) = std::numeric_limits<int64_t>::max();
  }
  LOG(INFO) << "Device size check: " << numRecords << " records, "
            << numUnsourced << " with a symbol not yet on the device, "
            << numCapacity << " capacities, " << numSkipped
            << " without a derived shape or shaped by host ops, "
            << found.sources.size() << " symbol sources";
  LOG(INFO) << "Device size check: symbols not yet on the device:"
            << largestCounts(unsourcedBySymbol, 30);
  LOG(INFO) << "Device alloc: region members by real size:"
            << largestCounts(regionRealWhy, 6);
  LOG(INFO) << "Device size check: members without a device size:"
            << largestCounts(memberNoSizeWhy, 12);
  LOG(INFO) << "Device size check: " << numPlaced << " group members placed, "
            << numUnplaced
            << " groups cut short by a member without a device size";
  LOG(INFO) << "Device size check: capacities without a symbolic form, by "
               "producer:"
            << largestCounts(noCapacityByProducer, 30);
  LOG(INFO) << "Device setup coverage: " << numCovered << " of "
            << numCoverSteps
            << " steps set up without host words; parameters keeping a host "
               "word:"
            << largestCounts(uncoveredByReason, 10)
            << "; by producer:" << largestCounts(unplacedByProducer, 20);
  // Host skip census: for each step whose parameters the setup writes
  // completely, whether the host reads its outputs, and if so whether the host
  // can compute their dims from graph input dims alone.
  {
    folly::F14FastSet<int32_t> readByUncovered;
    for (const auto& steps : state.stepVectors) {
      for (const auto& sv : steps) {
        if (sv.program == nullptr || sv.allWordsFromSetup ||
            (WaveConfig::get().deviceAlloc && sv.deviceWordsReady)) {
          continue;
        }
        for (const auto& fill : sv.program->fills) {
          readByUncovered.insert(fill.valueId);
        }
      }
    }
    int32_t numInternal = 0;
    int32_t numInternalFixed = 0;
    int32_t numHostComputable = 0;
    int32_t numCoveredSteps = 0;
    // For the device-sized path: steps whose groups all report a demand and
    // whose outputs all lie in groups, split by whether the host reads them.
    int32_t numDeviceSized = 0;
    int32_t numDeviceSizedInternal = 0;
    folly::F14FastMap<std::string, int64_t> notDeviceSizedWhy;
    folly::F14FastMap<std::string, int64_t> visibleWhy;
    std::vector<std::pair<int32_t, int32_t>> deviceSizedNotHostRead;
    for (size_t node = 0; node < state.stepVectors.size(); ++node) {
      const auto& steps = state.stepVectors[node];
      for (size_t step = 0; step < steps.size(); ++step) {
        const auto& sv = steps[step];
        if (sv.program == nullptr ||
            !(sv.allWordsFromSetup ||
              (WaveConfig::get().deviceAlloc && sv.deviceWordsReady))) {
          continue;
        }
        ++numCoveredSteps;
        const char* notSized = nullptr;
        auto planIt = planOf.find(static_cast<int32_t>(node));
        const auto* plan = planIt != planOf.end() ? planIt->second : nullptr;
        folly::F14FastSet<int32_t> ungroupedIds;
        if (plan != nullptr) {
          ungroupedIds.insert(plan->ungrouped.begin(), plan->ungrouped.end());
          if (step < plan->groupsByStep.size()) {
            for (auto g : plan->groupsByStep[step]) {
              const auto& group = plan->groups[g];
              auto slotIt = state.groupBaseSlot.find(&group);
              if (slotIt == state.groupBaseSlot.end() ||
                  state.groupHasDemand.at(slotIt->second) == 0) {
                notSized = notSized ? notSized
                    : group.concat != nullptr
                    ? "concat group without a device size"
                    : "group member without device size";
              }
            }
          }
        }
        for (const auto& launch : sv.program->launches) {
          for (const auto& bound : launch.outputs) {
            if (ungroupedIds.count(bound.valueId) != 0) {
              notSized = notSized ? notSized : "output outside any group";
            }
          }
        }
        bool internal = true;
        bool computable = true;
        bool hostReadAny = false;
        // Whether every output's memory size is fixed by the input dims and
        // the capacities: no data decides it.
        bool fixedSizes = true;
        for (const auto& launch : sv.program->launches) {
          for (const auto& bound : launch.outputs) {
            const auto id = bound.valueId;
            {
              const auto* shape = isSetOnDevice(id)
                  ? (capacityByName_.count(found.nameOf(id)) != 0
                         ? &capacityByName_.at(found.nameOf(id))
                         : nullptr)
                  : found.shapeOf(id);
              std::vector<std::string> sizeNames;
              if (shape != nullptr) {
                for (const auto& dim : shape->dims) {
                  dim.collectSymbols(sizeNames);
                }
              }
              fixedSizes =
                  fixedSizes && shape != nullptr &&
                  std::all_of(
                      sizeNames.begin(), sizeNames.end(), [&](const auto& n) {
                        return (!n.empty() && n[0] == 's') ||
                            picked.count(n) != 0;
                      });
            }
            const bool frameRead = id >= 0 &&
                id < static_cast<int32_t>(state.tableOnly.size()) &&
                state.tableOnly[id] == 0;
            const bool uncoveredRead = readByUncovered.count(id) != 0;
            // A graph output the host reads only at the end can be fetched,
            // and so can an input of standalones of later steps.
            hostReadAny = hostReadAny ||
                (frameRead &&
                 (static_cast<size_t>(id) >= state.outputOnly.size() ||
                  state.outputOnly[id] == 0) &&
                 (static_cast<size_t>(id) >= state.standaloneReadOnly.size() ||
                  state.standaloneReadOnly[id] == 0));
            if (!frameRead && !uncoveredRead) {
              continue;
            }
            internal = false;
            const auto* shape = found.shapeOf(id);
            std::vector<std::string> names;
            if (shape != nullptr) {
              symbolsOf(*shape, names);
            }
            // A symbol the host has: an input dim, or one the device returns
            // into the step's returned-value region, which the host reads.
            auto hostHas = [&](const std::string& n) {
              if (!n.empty() && n[0] == 's') {
                return true;
              }
              auto source = found.sources.find(n);
              if (source == found.sources.end() || source->second.node < 0 ||
                  static_cast<size_t>(source->second.node) >=
                      state.stepVectors.size() ||
                  static_cast<size_t>(source->second.step) >=
                      state.stepVectors[source->second.node].size()) {
                return false;
              }
              const auto& writer =
                  state.stepVectors[source->second.node][source->second.step];
              const auto offset =
                  source->second.offset & ~kDimsSource & 0xffffffffLL;
              return writer.program != nullptr &&
                  (source->second.offset & kElementSource) == 0 &&
                  writer.program->returnBegin >= 0 &&
                  offset >= writer.program->returnBegin &&
                  offset < writer.program->returnEnd;
            };
            const char* why = shape == nullptr ? "no derived shape"
                : !std::all_of(names.begin(), names.end(), hostHas)
                ? "dims not from input dims or returned values"
                : nullptr;
            if (why != nullptr) {
              computable = false;
              ++visibleWhy[fmt::format(
                  "{}, {}", frameRead ? "frame read" : "uncovered step", why)];
            }
          }
        }
        numInternal += internal ? 1 : 0;
        numInternalFixed += internal && fixedSizes ? 1 : 0;
        numHostComputable += !internal && computable ? 1 : 0;
        if (notSized != nullptr) {
          ++notDeviceSizedWhy[notSized];
          getStepVectors(
              state.stepVectors,
              static_cast<int32_t>(node),
              static_cast<int32_t>(step))
              .censusReason = notSized;
        } else {
          ++numDeviceSized;
          numDeviceSizedInternal += internal ? 1 : 0;
          if (!hostReadAny) {
            deviceSizedNotHostRead.emplace_back(
                static_cast<int32_t>(node), static_cast<int32_t>(step));
          } else {
            getStepVectors(
                state.stepVectors,
                static_cast<int32_t>(node),
                static_cast<int32_t>(step))
                .censusReason = "host reads an output";
          }
        }
      }
    }
    LOG(INFO) << "Device alloc census: of " << numCoveredSteps
              << " fully set up steps, " << numDeviceSized
              << " have every group's demand from the setup and every output "
                 "in a group; "
              << numDeviceSizedInternal
              << " of these have outputs only the device reads; the rest:"
              << largestCounts(notDeviceSizedWhy, 10);
    if (WaveConfig::get().deviceAlloc) {
      // A region member's shape at the capacities over the symbols the host
      // knows at a fetch: picked ones and input dims.
      auto hostShapeOf =
          [&](int32_t id) -> std::optional<ExecutionState::HostFetchShape> {
        const auto* shape = found.shapeOf(id);
        if (shape == nullptr || shape->dims.size() > kMaxDims ||
            shape->strides.size() != shape->dims.size()) {
          return std::nullopt;
        }
        bool known = true;
        std::function<SymExpr(const SymExpr&)> rename =
            [&](const SymExpr& expr) {
              if (expr.kind == SymExpr::Kind::kSymbol) {
                if (picked.count(expr.symbol) != 0) {
                  return SymExpr{
                      .kind = SymExpr::Kind::kSymbol,
                      .symbol =
                          "c" + std::to_string(symbolIndex_.at(expr.symbol))};
                }
                auto source = found.sources.find(expr.symbol);
                if (source != found.sources.end() &&
                    source->second.node == kInputSource) {
                  return SymExpr{
                      .kind = SymExpr::Kind::kSymbol,
                      .symbol = "i" + std::to_string(source->second.offset)};
                }
                known = false;
                return expr;
              }
              auto result = expr;
              for (auto& arg : result.args) {
                arg = rename(arg);
              }
              return result;
            };
        ExecutionState::HostFetchShape result;
        for (size_t d = 0; d < shape->dims.size(); ++d) {
          result.dims.push_back(rename(shape->dims[d]));
          result.strides.push_back(rename(shape->strides[d]));
        }
        if (!known) {
          return std::nullopt;
        }
        return result;
      };
      selectDeviceAllocSteps(
          state,
          planOf,
          deviceSizedNotHostRead,
          [&](int32_t id) { return found.shapeOf(id); },
          hostShapeOf,
          [&](int32_t id) {
            auto it = found.idToValue.find(id);
            if (it == found.idToValue.end() || it->second == nullptr) {
              return std::string("unknown");
            }
            const auto* producer = it->second->producer();
            if (producer == nullptr) {
              return std::string("input");
            }
            if (producer->target() != "tw.placeholder") {
              return std::string(producer->target());
            }
            // A kernel variant's value: its producer is in the variant graph.
            for (const auto& variant : waveGraph_.variantGraphs()) {
              for (const auto& node : variant->nodes()) {
                for (const auto* output : node.outputs()) {
                  if (output != nullptr && output->id() == id) {
                    return "variant " + std::string(node.target());
                  }
                }
              }
            }
            auto duplicate = waveGraph_.duplicates().find(id);
            if (duplicate != waveGraph_.duplicates().end()) {
              const auto original = duplicate->second.original;
              const auto& constraints = waveGraph_.types().constraints;
              return fmt::format(
                  "duplicate of %{} rank {}",
                  original,
                  static_cast<size_t>(original) < constraints.size()
                      ? constraints[original].rank
                      : -2);
            }
            return fmt::format("tw.placeholder %{}", id);
          });
    }
    LOG(INFO) << "Host skip census: of " << numCoveredSteps
              << " fully set up steps, " << numInternal
              << " have outputs only the device reads (" << numInternalFixed
              << " of them with sizes fixed by input dims and capacities), "
              << numHostComputable
              << " more have host-read outputs with dims from input dims or "
                 "returned values; "
                 "host-read outputs with other dims:"
              << largestCounts(visibleWhy, 10);
  }
  // Host-only census: the standalone nodes by target, where their inputs
  // come from and what reads their outputs; then the allocated values no
  // allocation group holds.
  {
    folly::F14FastSet<NodeCP> fusedNodes;
    // Every node a composite op maps, kernel or standalone.
    folly::F14FastSet<NodeCP> mappedNodes;
    // Per standalone node: whether it runs as a metadata shortcut, or is
    // metadata-only.
    folly::F14FastMap<NodeCP, const char*> standaloneKind;
    for (const auto& node : waveGraph_.nodes()) {
      const auto* invocation = node->kernels();
      if (invocation == nullptr) {
        continue;
      }
      for (const auto& op : invocation->ops()) {
        const auto& nodeMap = op.nodeMap();
        auto actualOf = [&](NodeCP formal) {
          auto it = nodeMap.find(formal);
          return it != nodeMap.end() ? it->second : formal;
        };
        for (const auto& [formal, actual] : nodeMap) {
          mappedNodes.insert(actual);
        }
        auto* projectOp = op.projectOp();
        for (auto* grid :
             {&projectOp->grid(),
              &projectOp->singleBlockGrid(),
              &projectOp->cgGrid()}) {
          for (const auto& step : *grid) {
            for (const auto& launch : step) {
              if (launch.op != nullptr) {
                for (auto* formal : launch.op->allNodes()) {
                  fusedNodes.insert(actualOf(formal));
                }
              } else if (launch.standalone != nullptr) {
                standaloneKind[actualOf(launch.standalone)] =
                    launch.standaloneShortcut != StandaloneShortcut::kNone
                    ? "shortcut"
                    : launch.metadataOnly ? "metadata"
                                          : "compute";
              }
            }
          }
        }
      }
    }
    auto shortTarget = [](std::string_view target) {
      constexpr std::string_view kPrefix = "torch.ops.";
      if (target.substr(0, kPrefix.size()) == kPrefix) {
        target.remove_prefix(kPrefix.size());
      }
      return std::string(target);
    };
    auto classOf = [&](const nativert::Node* node) -> std::string {
      if (node == nullptr) {
        return "none";
      }
      const auto target = node->target();
      if (target == "prim.Input") {
        return "graph input";
      }
      if (target == "prim.Output") {
        return "graph output";
      }
      auto it = standaloneKind.find(node);
      if (it != standaloneKind.end()) {
        return fmt::format("{} {}", it->second, shortTarget(target));
      }
      if (fusedNodes.count(node) != 0 || mappedNodes.count(node) != 0) {
        return "kernel";
      }
      return fmt::format("other {}", shortTarget(target));
    };
    struct TargetCensus {
      int32_t count{0};
      folly::F14FastMap<std::string, int64_t> inputs;
      folly::F14FastMap<std::string, int64_t> users;
    };
    std::map<std::string, TargetCensus> byTarget;
    for (const auto& [node, kind] : standaloneKind) {
      auto& census =
          byTarget[fmt::format("{} ({})", shortTarget(node->target()), kind)];
      ++census.count;
      for (const auto& input : node->inputs()) {
        ++census.inputs[classOf(input.value->producer())];
      }
      for (const auto* output : node->outputs()) {
        if (output == nullptr) {
          continue;
        }
        for (const auto* user : output->users()) {
          ++census.users[classOf(user)];
        }
      }
    }
    std::string text;
    for (const auto& [target, census] : byTarget) {
      text += fmt::format(
          "\n  {} x{}: inputs from{}; read by{}",
          target,
          census.count,
          largestCounts(census.inputs, 6),
          largestCounts(census.users, 6));
    }
    // Why each compute standalone is not fused: the registry's reason, and its
    // inputs' kinds, ranks and dtypes.
    std::string computeText;
    auto& types = waveGraph_.types();
    for (const auto& [node, kind] : standaloneKind) {
      if (std::string_view(kind) != "compute") {
        continue;
      }
      const auto* meta = Registry::metadata(node->target());
      std::string why = meta == nullptr ? "no registry entry"
          : meta->isStandalone_         ? "registered standalone"
                                        : "rule";
      if (meta != nullptr && !meta->isStandalone_) {
        bool rankRule = false;
        if (meta->only1d) {
          for (const auto& input : node->inputs()) {
            if (input.value != nullptr &&
                input.value->type().kind() == nativert::Type::Kind::Tensor) {
              const auto r = types.rank(input.value);
              rankRule = rankRule || r < 0 || r > 1;
            }
          }
        }
        why = rankRule               ? "only1d with rank > 1 or unknown"
            : meta->isStandaloneFunc ? "isStandaloneFunc"
                                     : "unknown";
      }
      std::string args;
      for (const auto& input : node->inputs()) {
        if (input.value == nullptr) {
          continue;
        }
        const auto* value = input.value;
        std::string desc = fmt::format(
            "{}:kind {} from {}",
            input.name,
            static_cast<int>(value->type().kind()),
            value->producer() != nullptr
                ? shortTarget(value->producer()->target())
                : std::string("none"));
        if (value->type().kind() == nativert::Type::Kind::Tensor) {
          const auto id = value->id();
          const auto* tensorMeta =
              id >= 0 && static_cast<size_t>(id) < types.types.size()
              ? types.types[id]
              : nullptr;
          desc = fmt::format(
              "{}:tensor rank {} {}",
              input.name,
              types.rank(value),
              tensorMeta != nullptr ? c10::toString(tensorMeta->dtype()) : "?");
        }
        args += (args.empty() ? "" : ", ") + desc;
      }
      std::string attrs;
      for (const auto& attribute : node->attributes()) {
        attrs += (attrs.empty() ? "" : ", ") + attribute.name + "=" +
            constantToString(attribute.value);
      }
      computeText += fmt::format(
          "\n  %{} {}: {}; inputs {}; attributes {}",
          node->outputs().empty() || node->outputs()[0] == nullptr
              ? -1
              : node->outputs()[0]->id(),
          shortTarget(node->target()),
          why,
          args,
          attrs);
    }
    LOG(INFO) << "Compute standalones:" << computeText;
    folly::F14FastMap<std::string, int64_t> unmapped;
    for (const auto& node : waveGraph_.graph()->nodes()) {
      const auto nodeClass = classOf(&node);
      if (nodeClass.rfind("other ", 0) == 0) {
        ++unmapped[nodeClass.substr(6)];
      }
    }
    LOG(INFO) << "Host-only census: " << standaloneKind.size()
              << " standalone nodes" << text
              << "\n  nodes in no composite op:" << largestCounts(unmapped, 25);
    folly::F14FastMap<std::string, int64_t> ungroupedBy;
    int32_t numUngrouped = 0;
    for (const auto& node : waveGraph_.nodes()) {
      const auto* invocation = node->kernels();
      const auto* plan =
          invocation != nullptr ? invocation->allocGroupPlan() : nullptr;
      if (plan == nullptr) {
        continue;
      }
      for (const auto id : plan->ungrouped) {
        ++numUngrouped;
        auto it = waveGraph_.idToValue().find(id);
        if (it == waveGraph_.idToValue().end()) {
          ++ungroupedBy["no value"];
          continue;
        }
        std::string readers;
        for (const auto* user : it->second->users()) {
          const auto userClass = classOf(user);
          if (readers.find(userClass) == std::string::npos) {
            readers += (readers.empty() ? "" : ", ") + userClass;
          }
        }
        ++ungroupedBy[fmt::format(
            "made by {}, read by {}",
            classOf(it->second->producer()),
            readers.empty() ? std::string("nothing") : readers)];
      }
    }
    LOG(INFO) << "Ungrouped census: " << numUngrouped
              << " allocated values in no group:"
              << largestCounts(ungroupedBy, 20) << "\n"
              << waveGraph_.allocGroupReport();
  }
  LOG(INFO) << "Device setup coverage: " << numAllWords << " of "
            << numCoverSteps
            << " steps with every parameter word from the setup; in covered "
               "steps, other words keeping a host word:"
            << largestCounts(otherByKind, 10);
}

void DeviceSymbols::poisonSizedParams(
    const SymbolFixups& fixups,
    uint8_t* params) {
  for (int32_t r = 0; r < fixups.numRecords; ++r) {
    const auto* record = fixups.records.data() + r * kRecordWords;
    if (record[1] == kPlacedRecord || record[1] == kPlacementRecord) {
      reinterpret_cast<Tensor*>(params + record[0])->storage = nullptr;
      continue;
    }
    // A word the setup resets or a scalar it computes gets a value the
    // kernels would misread.
    if (record[1] == kVolatileRecord) {
      if (record[2] == 0) {
        *reinterpret_cast<int64_t*>(params + record[0]) = 0x5a5a5a5a5a5a5a5a;
      } else if (record[2] == 1) {
        *reinterpret_cast<int32_t*>(params + record[0]) = 0x5a5a5a5a;
      } else {
        reinterpret_cast<Tensor*>(params + record[0])->status = Tensor::kInited;
      }
      continue;
    }
    if (record[1] == kScalarRecord) {
      *reinterpret_cast<int64_t*>(params + record[0]) = int64_t{1} << 40;
      continue;
    }
    // Check records are poisoned too: their checks then show where the
    // kernels read the host's dims.
    if (record[1] != kTensorRecord && record[1] != kShapeRecord &&
        record[1] != kSizedTensorRecord && record[1] != kSizedShapeRecord &&
        record[1] != kCheckRecord) {
      continue;
    }
    const auto rank = record[2] & 0xff;
    bool allDims = true;
    for (int32_t d = 0; d < rank; ++d) {
      allDims = allDims && record[4 + d] >= 0;
    }
    if (!allDims) {
      continue;
    }
    auto* tensor = reinterpret_cast<Tensor*>(params + record[0]);
    for (int32_t d = 0; d < rank; ++d) {
      tensor->dims[d] = 0;
    }
    tensor->numEl = 0;
  }
}

void DeviceSymbols::writeInputSymbols(
    const nativert::ExecutionFrame& frame,
    ExecutionState& state) const {
  if (state.sizeCheckInputs.empty()) {
    return;
  }
  std::vector<int64_t> values;
  values.reserve(state.sizeCheckInputs.size());
  for (const auto& [valueId, element, dim] : state.sizeCheckInputs) {
    c10::IValue value = frame.getIValue(valueId);
    if (element >= 0) {
      value = value.toListRef()[element];
    }
    if (dim == kAddressDim) {
      TableTensor t;
      writeTensorFields(value.toTensor().unsafeGetTensorImpl(), &t);
      values.push_back(reinterpret_cast<int64_t>(t.storage));
      continue;
    }
    values.push_back(dim >= 0 ? value.toTensor().size(dim) : value.toInt());
  }
  state.sizeCheckInputValues.copy_(
      at::from_blob(
          values.data(),
          {static_cast<int64_t>(values.size())},
          at::TensorOptions().dtype(at::kLong)));
  state.inputSymbolValues = std::move(values);
}

void DeviceSymbols::reportSizeChecks(ExecutionState& state) const {
  if (!state.sizeCheckCounts.defined()) {
    return;
  }
  const auto counts = state.sizeCheckCounts.cpu();
  const auto* values = counts.data_ptr<int64_t>();
  if (values[0] + values[3] > 0) {
    LOG(INFO) << "Device size check: " << values[0] << " checked, " << values[1]
              << " differ, " << values[2]
              << " differ on a data-decided symbol, " << values[3]
              << " differ in rank, " << values[4] << " differ in strides, "
              << values[5] << " differ in strides on a data-decided symbol, "
              << values[6] << " capacities checked, " << values[7]
              << " capacities differ, " << values[8] << " placements checked, "
              << values[9] << " placements differ";
  }
  state.sizeCheckCounts.copy_(at::zeros({10}, at::kLong));
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
