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

#include "velox/experimental/wave/exec/Wave.h"
#include <iostream>
#include "velox/experimental/wave/exec/Vectors.h"

DEFINE_int32(
    wave_rows_per_thread,
    4,
    "Number of rows per thread in generated kernels");
DEFINE_bool(wave_timing, true, "Enable Wave perf timers");
DEFINE_bool(
    wave_print_time,
    false,
    "Enables printing times inside PrinTime guard.");

DEFINE_bool(
    wave_transfer_timing,
    false,
    "Enables measuring host to device transfer latency separet "
    "from wait time for compute");

DEFINE_bool(wave_trace_stream, false, "Enable trace of streams and drivers");

DEFINE_int32(
    wave_init_group_by_buckets,
    2048,
    "Initial buckets in group by hash table (4 slots/bucket)");

namespace facebook::velox::wave {

PrintTime::PrintTime(const char* title)
    : title_(title),
      start_(FLAGS_wave_print_time ? getCurrentTimeMicro() : 0) {}

PrintTime::~PrintTime() {
  if (FLAGS_wave_print_time) {
    std::cout << title_ << "=" << getCurrentTimeMicro() - start_ << " "
              << comment_ << std::endl;
  }
}

std::string WaveTime::toString() const {
  if (micros < 20) {
    return fmt::format("{} ({} clocks)", succinctNanos(micros * 1000), clocks);
  }
  return succinctNanos(micros * 1000);
}

void WaveStats::add(const WaveStats& other) {
  numWaves += other.numWaves;
  numKernels += other.numKernels;
  numThreadBlocks += other.numThreadBlocks;
  numPrograms += other.numPrograms;
  numThreads += other.numThreads;
  numSync += other.numSync;
  bytesToDevice += other.bytesToDevice;
  bytesToHost += other.bytesToHost;
  hostOnlyTime += other.hostOnlyTime;
  hostParallelTime += other.hostParallelTime;
  waitTime += other.waitTime;
  transferWaitTime += other.transferWaitTime;
  stagingTime += other.stagingTime;
}

void WaveStats::clear() {
  new (this) WaveStats();
}

std::string Value::toString() const {
  std::stringstream out;
  if (subfield) {
    out << "<F " << subfield->toString() << ">";
  } else {
    out << "<E " << expr->toString(1) << ">";
  }
  return out.str();
}

const SubfieldMap*& threadSubfieldMap() {
  thread_local const SubfieldMap* subfields;
  return subfields;
}

std::string definesToString(const DefinesMap* map) {
  std::stringstream out;
  for (const auto& [value, id] : *map) {
    out
        << (value.subfield ? value.subfield->toString()
                           : value.expr->toString(1));
    out << " = " << id->id << " (" << id->type->toString() << ")" << std::endl;
  }
  return out.str();
}

void OperatorStateMap::addIfNew(
    int32_t id,
    const std::shared_ptr<OperatorState>& state) {
  std::lock_guard<std::mutex> l(mutex);
  if (states.count(id) == 0) {
    states[id] = state;
  }
}

AbstractOperand* pathToOperand(
    const DefinesMap& map,
    std::vector<std::unique_ptr<common::Subfield::PathElement>>& path) {
  if (path.empty()) {
    return nullptr;
  }
  common::Subfield field(std::move(path));
  const auto subfieldMap = threadSubfieldMap();
  auto it = threadSubfieldMap()->find(field.toString());
  if (it == subfieldMap->end()) {
    return nullptr;
  }
  Value value(it->second.get());
  auto valueIt = map.find(value);
  path = std::move(field.path());
  if (valueIt == map.end()) {
    return nullptr;
  }
  return valueIt->second;
}

WaveVector* Executable::operandVector(OperandId id) {
  WaveVectorPtr* ptr = nullptr;
  if (outputOperands.contains(id)) {
    auto ordinal = outputOperands.ordinal(id);
    ptr = &output[ordinal];
  }
  if (localOperands.contains(id)) {
    auto ordinal = localOperands.ordinal(id);
    ptr = &intermediates[ordinal];
  }
  if (*ptr) {
    return ptr->get();
  }
  return nullptr;
}

WaveVector* Executable::operandVector(OperandId id, const TypePtr& type) {
  WaveVectorPtr* ptr = nullptr;
  if (outputOperands.contains(id)) {
    auto ordinal = outputOperands.ordinal(id);
    ptr = &output[ordinal];
  } else if (localOperands.contains(id)) {
    auto ordinal = localOperands.ordinal(id);
    ptr = &intermediates[ordinal];
  } else {
    VELOX_FAIL("No local/output operand found");
  }
  if (*ptr) {
    return ptr->get();
  }
  *ptr = WaveVector::create(type, waveStream->arena());
  return ptr->get();
}

WaveStream::~WaveStream() {
  // Wait for device side activity. Memory accessed from device is live until
  // the streams are deleted, so block here.
  for (auto& stream : streams_) {
    stream->wait();
  }
  for (auto& exe : executables_) {
    if (exe->releaser) {
      exe->releaser(exe);
    }
  }
  releaseStreamsAndEvents();
}

void WaveStream::releaseStreamsAndEvents() {
  for (auto& stream : streams_) {
    releaseStream(std::move(stream));
  }
  for (auto& event : allEvents_) {
    std::unique_ptr<Event> temp(event);
    releaseEvent(std::move(temp));
  }
  allEvents_.clear();
  streams_.clear();
  lastEvent_.clear();
  hostReturnEvent_ = nullptr;
  // Conditional nullability will be set by the source.
  std::fill(operandNullable_.begin(), operandNullable_.end(), true);
}

void WaveStream::setState(WaveStream::State state) {
  if (state == state_) {
    return;
  }
  WaveTime nowTime = WaveTime::now();
  switch (state_) {
    case State::kNotRunning:
      break;
    case State::kHost:
      stats_.hostOnlyTime += nowTime - start_;
      break;
    case State::kParallel:
      stats_.hostParallelTime += nowTime - start_;
      break;
    case State::kWait:
      stats_.waitTime += nowTime - start_;
      break;
  }
  start_ = nowTime;
  state_ = state;
  if (state_ == State::kWait) {
    ++stats_.numSync;
  }
}

std::mutex WaveStream::reserveMutex_;
std::vector<std::unique_ptr<Stream>> WaveStream::streamsForReuse_;
std::vector<std::unique_ptr<Event>> WaveStream::eventsForReuse_;
bool WaveStream::exitInited_{false};
std::unique_ptr<folly::CPUThreadPoolExecutor> WaveStream::copyExecutor_;
std::unique_ptr<folly::CPUThreadPoolExecutor> WaveStream::syncExecutor_;

folly::CPUThreadPoolExecutor* WaveStream::copyExecutor() {
  return getExecutor(copyExecutor_);
}

folly::CPUThreadPoolExecutor* WaveStream::syncExecutor() {
  return getExecutor(syncExecutor_);
}

folly::CPUThreadPoolExecutor* WaveStream::getExecutor(
    std::unique_ptr<folly::CPUThreadPoolExecutor>& ptr) {
  if (ptr) {
    return ptr.get();
  }
  std::lock_guard<std::mutex> l(reserveMutex_);
  if (!ptr) {
    ptr = std::make_unique<folly::CPUThreadPoolExecutor>(32);
  }
  return ptr.get();
}

Stream* WaveStream::newStream() {
  auto stream = streamFromReserve();
  auto id = streams_.size();
  stream->userData() = reinterpret_cast<void*>(id);
  auto result = stream.get();
  streams_.push_back(std::move(stream));
  lastEvent_.push_back(nullptr);
  return result;
}

// static
void WaveStream::clearReusable() {
  streamsForReuse_.clear();
  eventsForReuse_.clear();
}

// static
std::unique_ptr<Stream> WaveStream::streamFromReserve() {
  std::lock_guard<std::mutex> l(reserveMutex_);
  if (streamsForReuse_.empty()) {
    auto result = std::make_unique<Stream>();
    if (!exitInited_) {
      // Register handler for clearing resources after first call of API.
      exitInited_ = true;
      atexit(WaveStream::clearReusable);
    }

    return result;
  }
  auto item = std::move(streamsForReuse_.back());
  streamsForReuse_.pop_back();
  return item;
}

//  static
void WaveStream::releaseStream(std::unique_ptr<Stream>&& stream) {
  std::lock_guard<std::mutex> l(reserveMutex_);
  streamsForReuse_.push_back(std::move(stream));
}
Event* WaveStream::newEvent() {
  auto event = eventFromReserve();
  auto result = event.release();
  allEvents_.insert(result);
  return result;
}

// static
std::unique_ptr<Event> WaveStream::eventFromReserve() {
  std::lock_guard<std::mutex> l(reserveMutex_);
  if (eventsForReuse_.empty()) {
    return std::make_unique<Event>();
  }
  auto item = std::move(eventsForReuse_.back());
  eventsForReuse_.pop_back();
  return item;
}

//  static
void WaveStream::releaseEvent(std::unique_ptr<Event>&& event) {
  std::lock_guard<std::mutex> l(reserveMutex_);
  eventsForReuse_.push_back(std::move(event));
}

OperatorState* WaveStream::operatorState(int32_t id) {
  auto it = taskStateMap_->states.find(id);
  if (it != taskStateMap_->states.end()) {
    return it->second.get();
  }
  return nullptr;
}

std::shared_ptr<OperatorState> WaveStream::operatorStateShared(int32_t id) {
  auto it = taskStateMap_->states.find(id);
  if (it != taskStateMap_->states.end()) {
    return it->second;
  }
  return nullptr;
}

OperatorState* WaveStream::newState(ProgramState& init) {
  auto stateShared = init.create(*this);
  taskStateMap_->states[init.stateId] = stateShared;
  return stateShared.get();
}

void WaveStream::markHostOutputOperand(const AbstractOperand& op) {
  hostOutputOperands_.add(op.id);
  auto nullable = isNullable(op);
  auto alignment = WaveVector::alignment(op.type);
  hostReturnSize_ = bits::roundUp(hostReturnSize_, alignment);
  hostReturnSize_ += WaveVector::backingSize(op.type, numRows_, nullable);
}

void WaveStream::setReturnData(bool needStatus) {
  if (!needStatus && hostReturnSize_ == 0) {
    return;
  }
}

void WaveStream::resultToHost() {
  if (!hostReturnEvent_) {
    hostReturnEvent_ = newEvent();
  }
  auto numBlocks = bits::roundUp(numRows_, kBlockSize) / kBlockSize;
  int32_t statusBytes = bits::roundUp(sizeof(BlockStatus) * numBlocks, 8) +
      instructionStatusSize(instructionStatus_, numBlocks);
  if (!hostBlockStatus_ || hostBlockStatus_->size() < statusBytes) {
    hostBlockStatus_ = getSmallTransferArena().allocate<char>(statusBytes);
  }
  Stream* transferStream = streams_[0].get();
  if (streams_.size() > 1) {
    // If many events, queue up the transfer on the first after
    for (auto i = 1; i < streams_.size(); ++i) {
      lastEvent_[i]->wait(*transferStream);
    }
  }
  if (hostReturnDataUsed_ > 0) {
    transferStream->deviceToHostAsync(
        hostReturnData_->as<char>(),
        deviceReturnData_->as<char>(),
        hostReturnDataUsed_);
  }
  transferStream->deviceToHostAsync(
      hostBlockStatus_->as<char>(), deviceBlockStatus_, statusBytes);
  hostReturnEvent_->record(*transferStream);
}

Executable* WaveStream::executableByInstruction(
    const AbstractInstruction* instruction) {
  for (auto& exe : executables_) {
    if (exe->programShared != nullptr) {
      for (auto& i : exe->programShared->instructions()) {
        if (i.get() == instruction) {
          return exe.get();
        }
      }
    }
  }
  return nullptr;
}

namespace {
// Copies from pageable host to unified address. Multithreaded memcpy is
// probably best.
void copyData(std::vector<Transfer>& transfers) {
  // TODO: Put memcpys or ppieces of them on AsyncSource if large enough.
  for (auto& transfer : transfers) {
    ::memcpy(transfer.to, transfer.from, transfer.size);
  }
}
} // namespace

void Executable::startTransfer(
    OperandSet outputOperands,
    std::vector<WaveVectorPtr>&& outputVectors,
    std::vector<Transfer>&& transfers,
    WaveStream& waveStream) {
  auto exe = std::make_unique<Executable>();
  auto numBlocks = bits::roundUp(waveStream.numRows(), kBlockSize) / kBlockSize;
  exe->waveStream = &waveStream;
  exe->outputOperands = outputOperands;
  WaveStream::ExeLaunchInfo info;
  waveStream.exeLaunchInfo(*exe, numBlocks, info);
  exe->output = std::move(outputVectors);
  exe->transfers = std::move(transfers);
  exe->deviceData.push_back(waveStream.arena().allocate<char>(info.totalBytes));
  auto start = exe->deviceData[0]->as<char>();
  exe->operands = waveStream.fillOperands(*exe, start, info)[0];
  copyData(exe->transfers);
  auto* device = waveStream.device();
  waveStream.installExecutables(
      folly::Range(&exe, 1),
      [&](Stream* stream, folly::Range<Executable**> executables) {
        for (auto& transfer : executables[0]->transfers) {
          stream->prefetch(device, transfer.to, transfer.size);
          waveStream.stats().bytesToDevice += transfer.size;
        }
        waveStream.markLaunch(*stream, *executables[0]);
      });
}

std::unique_ptr<Executable> WaveStream::recycleExecutable(
    Program* program,
    int32_t numRows) {
  for (auto i = 0; i < executables_.size(); ++i) {
    if (executables_[i]->programShared.get() == program) {
      auto result = std::move(executables_[i]);
      result->stream = nullptr;
      executables_.erase(executables_.begin() + i);
      return result;
    }
  }
  return nullptr;
}

void WaveStream::installExecutables(
    folly::Range<std::unique_ptr<Executable>*> executables,
    std::function<void(Stream*, folly::Range<Executable**>)> launch) {
  folly::F14FastMap<
      OperandSet,
      std::vector<Executable*>,
      OperandSetHasher,
      OperandSetComparer>
      dependences;
  for (auto& exeUnique : executables) {
    executables_.push_back(std::move(exeUnique));
    auto exe = executables_.back().get();
    exe->waveStream = this;
    VELOX_CHECK(exe->stream == nullptr);
    OperandSet streamSet;
    exe->inputOperands.forEach([&](int32_t id) {
      auto* source = operandToExecutable_[id];
      VELOX_CHECK(source != nullptr);
      auto stream = source->stream;
      if (stream) {
        // Compute pending, mark depenedency.
        auto sid = reinterpret_cast<uintptr_t>(stream->userData());
        streamSet.add(sid);
      }
    });
    dependences[streamSet].push_back(exe);
    exe->outputOperands.forEach([&](int32_t id) {
      // The stream may have the same or different exe in place from a previous
      // launch.
      operandToExecutable_[id] = exe;
    });
  }

  // exes with no dependences go on a new stream. Streams with dependent compute
  // get an event. The dependent computes go on new streams that first wait for
  // the events.
  folly::F14FastMap<int32_t, Event*> streamEvents;
  for (auto& [ids, exeVector] : dependences) {
    folly::Range<Executable**> exes(exeVector.data(), exeVector.size());
    std::vector<Stream*> required;
    ids.forEach([&](int32_t id) { required.push_back(streams_[id].get()); });
    if (required.size() == 1) {
      launch(required[0], exes);
      continue;
    }
    if (required.empty()) {
      Stream* stream = nullptr;
      Event* event = nullptr;
      for (auto i = 0; i < streams_.size(); ++i) {
        if (Stream* candidate = streams_[i].get()) {
          VELOX_CHECK_GT(lastEvent_.size(), i);
          if (!lastEvent_[i] || lastEvent_[i]->query()) {
            stream = candidate;
            event = lastEvent_[i];
            break;
          }
        }
      }
      if (!stream) {
        stream = newStream();
      }
      launch(stream, exes);
      if (event) {
        event->record(*stream);
      }
    } else {
      for (auto* req : required) {
        auto id = reinterpret_cast<uintptr_t>(req->userData());
        if (streamEvents.count(id) == 0) {
          auto event = newEvent();
          lastEvent_[id] = event;
          event->record(*req);
          streamEvents[id] = event;
        }
      }
      auto launchStream = newStream();
      ids.forEach([&](int32_t id) { streamEvents[id]->wait(*launchStream); });
      launch(launchStream, exes);
    }
  }
}

bool WaveStream::isArrived(
    const OperandSet& ids,
    int32_t sleepMicro,
    int32_t timeoutMicro) {
  OperandSet waitSet;
  if (hostReturnEvent_) {
    return hostReturnEvent_->query();
  }
  ids.forEach([&](int32_t id) {
    auto exe = operandToExecutable_[id];
    VELOX_CHECK_NOT_NULL(exe, "No exe produces operand {} in stream", id);
    if (!exe->stream) {
      return;
    }
    auto streamId = reinterpret_cast<uintptr_t>(exe->stream->userData());
    if (!lastEvent_[streamId]) {
      lastEvent_[streamId] = newEvent();
      lastEvent_[streamId]->record(*exe->stream);
    }
    if (lastEvent_[streamId]->query()) {
      return;
    }
    waitSet.add(streamId);
  });
  if (waitSet.empty()) {
    releaseStreamsAndEvents();
    return true;
  }
  if (sleepMicro == -1) {
    return false;
  }
  auto start = getCurrentTimeMicro();
  int64_t elapsed = 0;
  while (timeoutMicro == 0 || elapsed < timeoutMicro) {
    bool ready = true;
    waitSet.forEach([&](int32_t id) {
      if (!lastEvent_[id]->query()) {
        ready = false;
      }
    });
    if (ready) {
      releaseStreamsAndEvents();
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(sleepMicro));
    elapsed = getCurrentTimeMicro() - start;
  }
  return false;
}

void WaveStream::ensureVector(
    const AbstractOperand& op,
    WaveVectorPtr& vector,
    int32_t numRows) {
  if (!vector) {
    vector = std::make_unique<WaveVector>(op.type, arena());
  }
  bool nullable = isNullable(op);
  if (false /*hostOutputOperands_.contains(op.id)*/) {
    VELOX_NYI();
  } else {
    vector->resize(numRows < 0 ? numRows_ : numRows, nullable);
  }
}

bool WaveStream::isNullable(const AbstractOperand& op) const {
  if (op.notNull) {
    return false;
  }
  if (op.sourceNullable) {
    return operandNullable_[op.id];
  }
  if (op.conditionalNonNull) {
    for (auto i : op.nullableIf) {
      if (operandNullable_[i]) {
        return true;
      }
    }
    return false;
  }
  return true;
}

void WaveStream::exeLaunchInfo(
    Executable& exe,
    int32_t numBlocks,
    ExeLaunchInfo& info) {
  // The exe has an Operand* for each input/local/output/literal
  // op. It has an Operand for each local/output/literal op. It has
  // an array of numBlock int32_t*'s for every distinct wrapAt in
  // its local/output operands where the wrapAt does not occur in
  // any of the input Operands.
  info.numBlocks = numBlocks;
  info.numInput = exe.inputOperands.size();
  exe.inputOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    auto* inputExe = operandExecutable(op->id);
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      auto* indices = inputExe->wraps[op->wrappedAt];
      VELOX_CHECK_NOT_NULL(indices);
      info.inputWrap[op->wrappedAt] = indices;
    }
  });

  exe.localOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      if (info.inputWrap.find(id) == info.inputWrap.end()) {
        if (info.localWrap.find(op->wrappedAt) == info.localWrap.end()) {
          info.localWrap[op->wrappedAt] = reinterpret_cast<int32_t**>(
              info.localWrap.size() * numBlocks * sizeof(void*));
        }
      }
    }
  });
  exe.outputOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      if (info.inputWrap.find(id) == info.inputWrap.end()) {
        if (info.localWrap.find(op->wrappedAt) == info.localWrap.end()) {
          info.localWrap[op->wrappedAt] = reinterpret_cast<int32_t**>(
              info.localWrap.size() * numBlocks * sizeof(void*));
        }
      }
    }
  });
  if (exe.programShared) {
    info.numExtraWrap = exe.programShared->extraWrap().size();
    exe.programShared->extraWrap().forEach([&](auto id) {
      auto op = operandAt(id);
      auto* inputExe = operandExecutable(op->id);
      VELOX_CHECK(op->wrappedAt != AbstractOperand::kNoWrap);
      auto* indices = inputExe->wraps[op->wrappedAt];
      VELOX_CHECK_NOT_NULL(indices);
    });
  }
  info.numLocalOps = exe.localOperands.size() + exe.outputOperands.size();
  info.totalBytes =
      // Pointer to Operand for input and local Operands and extra wraps.
      sizeof(void*) *
          (info.numLocalOps + exe.inputOperands.size() + info.numExtraWrap) +
      // Flat array of Operand for all but input and extra wrap.
      sizeof(Operand) * info.numLocalOps +
      // Space for the 'indices' for each distinct wrappedAt.
      (info.localWrap.size() * numBlocks * sizeof(void*));
  if (exe.programShared) {
    exe.programShared->getOperatorStates(*this, info.operatorStates);
  }
}

Operand**
WaveStream::fillOperands(Executable& exe, char* start, ExeLaunchInfo& info) {
  Operand** operandPtrBegin = addBytes<Operand**>(start, 0);
  auto initialOperandBegin = operandPtrBegin;
  exe.inputOperands.forEach([&](int32_t id) {
    auto* inputExe = operandToExecutable_[id];
    int32_t ordinal = inputExe->outputOperands.ordinal(id);
    *operandPtrBegin =
        &inputExe->operands[inputExe->firstOutputOperandIdx + ordinal];
    ++operandPtrBegin;
  });
  Operand* operandBegin = addBytes<Operand*>(
      start,
      (info.numInput + info.numLocalOps + info.numExtraWrap) * sizeof(void*));
  VELOX_CHECK_EQ(0, reinterpret_cast<uintptr_t>(operandBegin) & 7);
  int32_t* indicesBegin =
      addBytes<int32_t*>(operandBegin, info.numLocalOps * sizeof(Operand));
  for (auto& [id, ptr] : info.localWrap) {
    info.localWrap[id] =
        addBytes<int32_t**>(indicesBegin, reinterpret_cast<int64_t>(ptr));
  }
  exe.wraps = std::move(info.localWrap);
  for (auto& [id, ptr] : info.inputWrap) {
    exe.wraps[id] = ptr;
  }
  exe.intermediates.resize(exe.localOperands.size());
  int32_t fill = 0;
  exe.localOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    ensureVector(*op, exe.intermediates[fill]);
    auto vec = exe.intermediates[fill].get();
    ++fill;
    vec->toOperand(operandBegin);
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      operandBegin->indices = exe.wraps[op->wrappedAt];
      VELOX_CHECK_NOT_NULL(operandBegin->indices);
    }
    *operandPtrBegin = operandBegin;
    ++operandPtrBegin;
    ++operandBegin;
  });
  exe.firstOutputOperandIdx = exe.intermediates.size();
  exe.output.resize(exe.outputOperands.size());
  fill = 0;
  exe.outputOperands.forEach([&](auto id) {
    auto op = operandAt(id);
    ensureVector(*op, exe.output[fill]);
    auto vec = exe.output[fill].get();
    ++fill;
    vec->toOperand(operandBegin);
    if (op->wrappedAt != AbstractOperand::kNoWrap) {
      operandBegin->indices = exe.wraps[op->wrappedAt];
      VELOX_CHECK_NOT_NULL(operandBegin->indices);
    }
    *operandPtrBegin = operandBegin;
    ++operandPtrBegin;
    ++operandBegin;
  });

  info.firstExtraWrap = operandPtrBegin - initialOperandBegin;
  if (exe.programShared) {
    exe.programShared->extraWrap().forEach([&](int32_t id) {
      auto* inputExe = operandToExecutable_[id];
      int32_t ordinal = inputExe->outputOperands.ordinal(id);
      *operandPtrBegin =
          &inputExe->operands[inputExe->firstOutputOperandIdx + ordinal];
      ++operandPtrBegin;
    });
  }
  return addBytes<Operand**>(start, 0);
}

void WaveStream::setLaunchControl(
    int32_t key,
    int32_t nth,
    std::unique_ptr<LaunchControl> control) {
  if (key == 0 && nth == 0) {
    deviceBlockStatus_ = control->params.status;
  }
  auto& controls = launchControl_[key];
  if (controls.size() <= nth) {
    controls.resize(nth + 1);
  }
  controls[nth] = std::move(control);
}

LaunchControl* WaveStream::prepareProgramLaunch(
    int32_t key,
    int32_t nthLaunch,
    int32_t inputRows,
    folly::Range<Executable**> exes,
    int32_t inputBlocksPerExe,
    const LaunchControl* inputControl,
    Stream* stream) {
  static_assert(Operand::kPointersInOperand * sizeof(void*) == sizeof(Operand));
  auto rowsPerThread = FLAGS_wave_rows_per_thread;
  int32_t blocksPerExe = bits::roundUp(inputBlocksPerExe, rowsPerThread);
  auto& controlVector = launchControl_[key];
  LaunchControl* controlPtr;
  if (controlVector.size() > nthLaunch) {
    controlPtr = controlVector[nthLaunch].get();
  } else {
    controlVector.resize(nthLaunch + 1);
    controlVector[nthLaunch] = std::make_unique<LaunchControl>(key, inputRows);
    controlPtr = controlVector[nthLaunch].get();
  }
  // tru if not first launch.
  bool isContinue = false;
  // true if redoing selected lanes of a previous launch. Requires using the
  // same control block as in the previous launch.
  bool isRetry = false;
  auto& control = *controlPtr;
  if (control.programInfo.empty()) {
    control.programInfo.resize(exes.size());
  } else {
    VELOX_CHECK_EQ(exes.size(), control.programInfo.size());
    for (auto& info : control.programInfo) {
      if (info.advance.empty()) {
        continue;
      }
      isContinue = true;
      if (info.advance.isRetry) {
        isContinue = true;
        isRetry = true;
        checkBlockStatuses();
        break;
      } else {
        numRows_ = info.advance.numRows;
        inputBlocksPerExe = bits::roundUp(numRows_, kBlockSize) / kBlockSize;
        blocksPerExe = bits::roundUp(inputBlocksPerExe, rowsPerThread);
      }
    }
  }
  if (isContinue) {
    VELOX_CHECK_EQ(-1, inputRows);
  } else {
    VELOX_CHECK_LT(0, inputRows);
    numRows_ = inputRows;
  }
  int32_t numBranches = 1;
  if (!exes.empty() && exes[0]->programShared->kernel()) {
    numBranches = exes[0]->programShared->numBranches();
    VELOX_CHECK_EQ(1, exes.size());
  }
  // 2 int arrays: blockBase, programIdx.
  int32_t numBlocks =
      std::max<int32_t>(1, exes.size()) * blocksPerExe * numBranches;
  int32_t size = 2 * numBlocks * sizeof(int32_t);
  std::vector<ExeLaunchInfo> info(exes.size());
  // 2 pointers per exe: TB program and start of its param array and 1 int for
  // start PC. Round to 3 for alignment.
  size += exes.size() * sizeof(void*) * 3;
  auto operandOffset = size;
  // Exe dependent sizes for operands.
  int32_t operandBytes = 0;
  int32_t operatorStateBytes = 0;
  int32_t shared = 0;
  for (auto i = 0; i < exes.size(); ++i) {
    exeLaunchInfo(*exes[i], numBlocks, info[i]);
    operandBytes += info[i].totalBytes;
    markLaunch(*stream, *exes[i]);
    shared = std::max(shared, exes[i]->programShared->sharedMemorySize());
    operatorStateBytes += info[i].operatorStates.size() * sizeof(void*);
  }
  size += operandBytes;
  int32_t statusOffset = 0;
  int32_t statusBytes = 0;
  if (!inputControl) {
    statusOffset = size;
    //  Pointer to return block for each tB.
    statusBytes = bits::roundUp(blocksPerExe * sizeof(BlockStatus), 8);
    statusBytes += bits::roundUp(
        instructionStatus_.gridStateSize +
            instructionStatus_.blockState * numBlocks,
        8);
    size += statusBytes;
  }
  // 1 pointer per exe and an exe-dependent data area.
  int32_t operatorStateOffset = size;
  size += exes.size() * sizeof(void*) + operatorStateBytes;
  WaveBufferPtr buffer;
  if (isRetry) {
    buffer = std::move(control.deviceData);
  } else {
    buffer = arena_->allocate<char>(size);
  }
  if (stream) {
    stream->prefetch(nullptr, buffer->as<char>(), buffer->size());
  }
  // Zero initialization is expected, for example for operands and arrays in
  // Operand::indices.
  if (!isRetry) {
    memset(buffer->as<char>(), 0, size);
  }
  control.sharedMemorySize = shared;
  // Now we fill in the various arrays and put their start addresses in
  // 'control'.
  auto start = buffer->as<int32_t>();
  control.params.blockBase = start;
  control.params.programIdx = start + numBlocks;
  control.params.numRowsPerThread = FLAGS_wave_rows_per_thread;
  control.params.operands = addBytes<Operand***>(
      control.params.programIdx, numBlocks * sizeof(int32_t));
  control.params.startPC = isContinue
      ? addBytes<int32_t*>(control.params.operands, exes.size() * sizeof(void*))
      : nullptr;

  if (!inputControl && !isRetry) {
    // If the launch produces new statuses (as opposed to updating status of a
    // previous launch), there is an array with a status for each TB. If there
    // are multiple exes, they all share the same error codes. A launch can have
    // a single cardinality change, which will update the row counts in each TB.
    // Writing errors is not serialized but each lane with at least one error
    // will show one error.
    control.params.status = addBytes<BlockStatus*>(start, statusOffset);
    deviceBlockStatus_ = control.params.status;
    // Memory is already set to all 0.
    for (auto i = 0; i < inputBlocksPerExe; ++i) {
      auto status = &control.params.status[i];
      status->numRows =
          i == inputBlocksPerExe - 1 ? inputRows % kBlockSize : kBlockSize;
    }
  } else if (!inputControl) {
    // No input control and retry. the statuses are as left by the previous try.
    ;
  } else {
    control.params.status = inputControl->params.status;
  }
  char* operandStart = addBytes<char*>(start, operandOffset);
  VELOX_CHECK_EQ(0, reinterpret_cast<uintptr_t>(operandStart) & 7);
  int32_t fill = 0;
  for (auto i = 0; i < exes.size(); ++i) {
    if (isContinue) {
      if (control.programInfo[i].advance.empty()) {
        control.params.startPC[i] = -1;
      } else {
        control.params.startPC[i] =
            control.programInfo[i].advance.continueLabel;
      }
    }
    auto operandPtrs = fillOperands(*exes[i], operandStart, info[i]);
    if (exes.size() == 1) {
      control.params.extraWraps = info[0].firstExtraWrap;
      control.params.numExtraWraps = exes[0]->programShared->extraWrap().size();
    }
    control.params.operands[i] = operandPtrs;
    // The operands defined by the exe start after the input operands and are
    // all consecutive.
    exes[i]->operands = operandPtrs[exes[i]->inputOperands.size()];
    operandStart += info[i].totalBytes;
    for (auto tbIdx = 0; tbIdx < blocksPerExe; ++tbIdx) {
      control.params.blockBase[fill] = i * blocksPerExe;
      control.params.programIdx[fill] = i;
      ++fill;
    }
  }

  // Fill in operator states, e.g. hash tables.
  void** operatorStatePtrs = addBytes<void**>(start, operatorStateOffset);
  control.params.operatorStates = reinterpret_cast<void***>(operatorStatePtrs);
  auto stateFill = operatorStatePtrs + info.size();
  for (auto i = 0; i < info.size(); ++i) {
    operatorStatePtrs[i] = stateFill;
    auto& ptrs = info[i].operatorStates;
    for (auto j = 0; j < ptrs.size(); ++j) {
      *stateFill = ptrs[j];
      ++stateFill;
    }
  }
  control.params.numBlocks = inputBlocksPerExe;
  control.params.streamIdx = streamIdx_;
  if (!exes.empty()) {
    ++stats_.numKernels;
  }

  stats_.numPrograms += exes.size();
  stats_.numThreadBlocks += blocksPerExe * exes.size();
  stats_.numThreads += numRows_ * exes.size();

  control.deviceData = std::move(buffer);
  return &control;
}

int32_t WaveStream::getOutput(
    int32_t operatorId,
    memory::MemoryPool& pool,
    folly::Range<const OperandId*> operands,
    VectorPtr* vectors) {
  checkExecutables();
  auto it = launchControl_.find(operatorId);
  VELOX_CHECK(it != launchControl_.end());
  auto* control = it->second[0].get();
  auto* status = control->params.status;
  auto numBlocks = bits::roundUp(numRows_, kBlockSize) / kBlockSize;
  if (operands.empty()) {
    return statusNumRows(status, numBlocks);
  }
  for (auto i = 0; i < operands.size(); ++i) {
    auto id = operands[i];
    auto exe = operandExecutable(id);
    VELOX_CHECK_NOT_NULL(exe);
    exe->stream->wait();
    auto ordinal = exe->outputOperands.ordinal(id);
    auto waveVectorPtr = &exe->output[ordinal];
    if (!waveVectorPtr->get()) {
      exe->ensureLazyArrived(operands);
      VELOX_CHECK_NOT_NULL(
          waveVectorPtr->get(), "Lazy load should have filled in the result");
    }
    vectors[i] = waveVectorPtr->get()->toVelox(
        &pool,
        numBlocks,
        status,
        &exe->operands[exe->firstOutputOperandIdx + ordinal]);
  }
  return vectors[0]->size();
}

void AggregateOperatorState::allocateAggregateHeader(
    int32_t size,
    GpuArena& arena) {
  // Size and alignment of page of unified memory. Swappable host-device at page
  // granularity.
  constexpr size_t kUnifiedPageSize = 4096;
  int32_t alignedSize =
      bits::roundUp(size, kUnifiedPageSize) + kUnifiedPageSize;
  WaveBufferPtr head = arena.allocate<char>(alignedSize);
  VELOX_CHECK(buffers.empty());
  buffers.push_back(head);
  auto address = reinterpret_cast<uintptr_t>(head->as<char>());
  alignedHead = reinterpret_cast<DeviceAggregation*>(
      bits::roundUp(address, kUnifiedPageSize));
  alignedHeadSize = size;
  new (alignedHead) DeviceAggregation();
}

void WaveStream::makeHashTable(
    AggregateOperatorState& state,
    int32_t rowSize,
    bool makeTable) {
  AggregationControl control;
  auto stream = streamFromReserve();
  const int32_t numPartitions = 1;
  int32_t size = sizeof(DeviceAggregation) + sizeof(GpuHashTableBase) +
      sizeof(HashPartitionAllocator) * numPartitions;
  state.allocateAggregateHeader(size, *arena_);
  auto* header = state.alignedHead;
  auto* hashTable = reinterpret_cast<GpuHashTableBase*>(header + 1);
  state.hashTable = hashTable;
  HashPartitionAllocator* allocators =
      reinterpret_cast<HashPartitionAllocator*>(hashTable + 1);
  int32_t numBuckets = bits::nextPowerOfTwo(FLAGS_wave_init_group_by_buckets);
  header->table = hashTable;
  WaveBufferPtr table;
  if (makeTable) {
    table = arena_->allocate<char>(sizeof(GpuBucketMembers) * numBuckets);
    state.buffers.push_back(table);
  }
  new (hashTable) GpuHashTableBase(
      makeTable ? table->as<GpuBucket>() : nullptr,
      numBuckets - 1,
      0,
      reinterpret_cast<RowAllocator*>(allocators));
  auto numRows = makeTable ? numBuckets * GpuBucketMembers::kNumSlots
                           : (1 << 20) / rowSize;
  WaveBufferPtr rows = arena_->allocate<char>(rowSize * numRows);
  state.buffers.push_back(rows);
  new (allocators) HashPartitionAllocator(
      rows->as<char>(), rows->size(), rows->size(), rowSize);
  state.setSizesToSafe();
  stream->prefetch(getDevice(), state.alignedHead, state.alignedHeadSize);
  if (table) {
    stream->memset(table->as<char>(), 0, table->size());
  }
  releaseStream(std::move(stream));
}

void WaveStream::makeAggregate(
    AbstractAggregation& inst,
    AggregateOperatorState& state) {
  AggregationControl control;
  auto stream = streamFromReserve();
  if (inst.keys.empty()) {
    int32_t size = inst.rowSize() + sizeof(DeviceAggregation);
    state.allocateAggregateHeader(size, *arena_);
    control.head = state.alignedHead;
    control.headSize = size;
    control.rowSize = inst.rowSize();
    reinterpret_cast<WaveKernelStream*>(stream.get())
        ->setupAggregation(control, 0, nullptr);
  } else {
    const int32_t numPartitions = 1;
    int32_t size = sizeof(DeviceAggregation) + sizeof(GpuHashTableBase) +
        sizeof(HashPartitionAllocator) * numPartitions;
    state.allocateAggregateHeader(size, *arena_);
    auto* header = state.alignedHead;
    auto* hashTable = reinterpret_cast<GpuHashTableBase*>(header + 1);
    HashPartitionAllocator* allocators =
        reinterpret_cast<HashPartitionAllocator*>(hashTable + 1);
    int32_t numBuckets = bits::nextPowerOfTwo(FLAGS_wave_init_group_by_buckets);
    header->table = hashTable;
    WaveBufferPtr table =
        arena_->allocate<char>(sizeof(GpuBucketMembers) * numBuckets);
    state.buffers.push_back(table);

    new (hashTable) GpuHashTableBase(
        table->as<GpuBucket>(),
        numBuckets - 1,
        0,
        reinterpret_cast<RowAllocator*>(allocators));
    auto rowSize = inst.rowSize();
    auto numRows = numBuckets * GpuBucketMembers::kNumSlots;
    WaveBufferPtr rows = arena_->allocate<char>(rowSize * numRows);
    state.buffers.push_back(rows);
    new (allocators) HashPartitionAllocator(
        rows->as<char>(), rows->size(), rows->size(), rowSize);
    state.setSizesToSafe();
    stream->prefetch(getDevice(), state.alignedHead, state.alignedHeadSize);
    stream->memset(table->as<char>(), 0, table->size());
  }
  releaseStream(std::move(stream));
}

void WaveStream::makeHashBuild(
    AbstractHashBuild& inst,
    HashTableHolder& state) {
  makeHashTable(state, inst.rowSize(), false);
}

void checkOperand(Operand& op) {
  if (op.indexMask != 0 && op.indexMask != -1) {
    VELOX_FAIL("Corrupt operand in executable");
  }
}

void WaveStream::checkExecutables() const {
  for (auto& pair : operandToExecutable_) {
    bool found = false;
    for (auto& exe : executables_) {
      if (exe.get() == pair.second) {
        found = true;
        break;
      }
    }
    if (!found) {
      VELOX_FAIL("Operand exe not found in owned executables");
    }
    auto* exe = pair.second;
    if (exe->operands) {
      auto numOperands =
          exe->firstOutputOperandIdx + exe->outputOperands.size();
      for (auto i = 0; i < numOperands; ++i) {
        auto& op = exe->operands[i];
        checkOperand(op);
      }
    }
  }
}

void WaveStream::throwIfError(std::function<void(const KernelError*)> action) {
  auto numBlocks = bits::roundUp(numRows_, kBlockSize) / kBlockSize;
  auto hostSide = hostBlockStatus();
  int32_t errorOffset = bits::roundUp(numBlocks * sizeof(BlockStatus), 8);
  auto error = addBytes<KernelError*>(hostSide, errorOffset);
  if (error->messageEnum) {
    setError();
    action(error);
  }
}

void WaveStream::checkBlockStatuses() const {
#ifdef BLOCK_STATUS_CHECK
  auto numBlocks = bits::roundUp(numRows_, kBlockSize) / kBlockSize;
  auto hostSide = hostBlockStatus();
  auto deviceSide = deviceBlockStatus_;
  for (auto i = 0; i < numBlocks; ++i) {
    for (auto j = 0; j < kBlockSize; ++j) {
      if (hostSide) {
        VELOX_CHECK_LE(hostSide[i].numRows, 256);
        VELOX_CHECK_LE(static_cast<uint8_t>(hostSide[i].errors[j]), 4);
      }
      if (deviceSide) {
        VELOX_CHECK_LE(deviceSide[i].numRows, 256);
        VELOX_CHECK_LE(static_cast<uint8_t>(deviceSide[i].errors[j]), 4);
      }
    }
  }
#endif
}

std::string WaveStream::toString() const {
  std::stringstream out;
  out << "{WaveStream ";
  for (auto& exe : executables_) {
    out << exe->toString() << std::endl;
  }
  out << "}";
  if (hostReturnEvent_) {
    out << fmt::format("hostReturnEvent={}", hostReturnEvent_->query())
        << std::endl;
  }
  for (auto i = 0; i < streams_.size(); ++i) {
    out << fmt::format(
        "stream {} {}, ",
        streams_[i]->userData(),
        lastEvent_[i] ? fmt::format("event={}", lastEvent_[i]->query())
                      : fmt::format("no event"));
  }
  return out.str();
}

WaveTypeKind typeKindCode(TypeKind kind) {
  return static_cast<WaveTypeKind>(kind);
}

Program::Program(
    OperandSet input,
    OperandSet local,
    OperandSet output,
    OperandSet extraWrap,
    int32_t numBranches,
    int32_t sharedSize,
    const std::vector<std::unique_ptr<AbstractOperand>>& allOperands,
    std::vector<std::unique_ptr<ProgramState>> states,
    std::unique_ptr<CompiledKernel> kernel)
    : kernel_(std::move(kernel)),
      outputIds_(output),
      extraWrap_(extraWrap),
      numBranches_(numBranches),
      sharedMemorySize_(sharedSize),
      operatorStates_(std::move(states)) {
  input.forEach([&](int32_t id) { input_[allOperands[id].get()] = id; });
  local.forEach([&](int32_t id) { local_[allOperands[id].get()] = id; });
  output.forEach([&](int32_t id) { output_[allOperands[id].get()] = id; });
}

void Program::getOperatorStates(WaveStream& stream, std::vector<void*>& ptrs) {
  ptrs.resize(operatorStates_.size());
  for (auto i = 0; i < operatorStates_.size(); ++i) {
    auto& operatorState = *operatorStates_[i];
    if (operatorState.isGlobal) {
      auto* taskStates = stream.taskStateMap();
      std::lock_guard<std::mutex> l(taskStates->mutex);
      auto* state = stream.operatorState(operatorState.stateId);
      if (!state) {
        VELOX_CHECK_NOT_NULL(operatorState.create);
        state = stream.newState(operatorState);
      }
      ptrs[i] = state->devicePtr();
    } else {
      auto* state = stream.operatorState(operatorState.stateId);
      if (!state) {
        VELOX_CHECK_NOT_NULL(operatorState.create);
        state = stream.newState(operatorState);
      }
      ptrs[i] = state->devicePtr();
    }
  }
}

bool Program::isSink() const {
  int32_t size = instructions_.size();
  return size > 0 && instructions_[size - 1]->isSink();
}

exec::BlockingReason Program::isBlocked(
    WaveStream& stream,
    ContinueFuture* future) {
  for (int32_t i = instructions_.size() - 1; i >= 0; --i) {
    auto* instruction = instructions_[i].get();
    OperatorState* state = nullptr;
    auto stateId = instruction->stateId();
    if (stateId.has_value()) {
      state = stream.operatorState(stateId.value());
    }
    auto result = instruction->isBlocked(stream, state, future);
    if (result != exec::BlockingReason::kNotBlocked) {
      return result;
    }
  }
  return exec::BlockingReason::kNotBlocked;
}

AdvanceResult Program::canAdvance(
    WaveStream& stream,
    LaunchControl* control,
    int32_t programIdx) {
  for (int32_t i = instructions_.size() - 1; i >= 0; --i) {
    auto* instruction = instructions_[i].get();
    OperatorState* state = nullptr;
    auto stateId = instruction->stateId();
    if (stateId.has_value()) {
      state = stream.operatorState(stateId.value());
    }
    auto result = instruction->canAdvance(stream, control, state, i);
    if (!result.empty()) {
      result.instructionIdx = i;
      result.programIdx = programIdx;
      return result;
    }
  }
  return {};
}

void Program::callUpdateStatus(
    WaveStream& stream,
    const std::vector<WaveStream*>& otherStreams,
    AdvanceResult& advance) {
  if (advance.updateStatus) {
    advance.updateStatus(
        stream, otherStreams, *instructions_[advance.instructionIdx]);
  }
}

void Program::pipelineFinished(WaveStream& stream) {
  for (auto& instruction : instructions_) {
    instruction->pipelineFinished(stream, this);
  }
}

std::unique_ptr<Executable> Program::getExecutable(
    int32_t maxRows,
    const std::vector<std::unique_ptr<AbstractOperand>>& operands) {
  std::unique_ptr<Executable> exe;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (!prepared_.empty()) {
      exe = std::move(prepared_.back());
      exe->programShared = shared_from_this();
      prepared_.pop_back();
    }
  }
  if (!exe) {
    exe = std::make_unique<Executable>();
    exe->programShared = shared_from_this();
    for (auto& pair : input_) {
      exe->inputOperands.add(pair.first->id);
    }
    for (auto& pair : local_) {
      exe->localOperands.add(pair.first->id);
    }
    for (auto& pair : output_) {
      exe->outputOperands.add(pair.first->id);
    }

    exe->releaser = [](std::unique_ptr<Executable>& ptr) {
      auto program = ptr->programShared.get();
      ptr->reuse();
      program->releaseExe(std::move(ptr));
    };
  }
  return exe;
}

std::string AbstractOperand::toString() const {
  if (constant) {
    return fmt::format(
        "<literal {} {}>", constant->toString(0), type->toString());
  }
  const char* nulls = notNull ? "NN"
      : conditionalNonNull    ? "CN"
      : sourceNullable        ? "SN"
                              : "?";
  return fmt::format("<{} {}: {} {}>", id, label, nulls, type->toString());
}

std::string Executable::toString() const {
  std::stringstream out;
  out << "{Exe "
      << (stream ? fmt::format("stream {}", stream->userData())
                 : fmt::format(" no stream "))
      << " produces ";
  bool first = true;
  outputOperands.forEach([&](auto id) {
    if (!first) {
      out << ", ";
    };
    first = false;
    out << waveStream->operandAt(id)->toString();
  });
  if (programShared) {
    out << std::endl;
    out << "program " << programShared->toString();
  }
  return out.str();
}

std::string Program::toString() const {
  std::stringstream out;
  out << "{ program" << std::endl;
  for (auto& instruction : instructions_) {
    out << instruction->toString() << std::endl;
  }
  out << "}" << std::endl;
  return out.str();
}

} // namespace facebook::velox::wave
