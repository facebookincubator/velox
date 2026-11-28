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
#include "velox/exec/python/PythonWorker.h"

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/util/key_value_metadata.h>
#include <folly/Conv.h>
#include <folly/FileUtil.h>
#include <folly/String.h>
#include <folly/container/F14Set.h>
#include <folly/json.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include "velox/exec/python/PythonEnvironmentResolver.h"
#include "velox/exec/python/WorkerScript.h"
#include "velox/vector/arrow/Bridge.h"

extern char** environ;

namespace facebook::velox::exec::python {
namespace {

constexpr int kProtocolVersion{1};

// Message types, as README.md defines them.
constexpr uint8_t kSetup{1};
constexpr uint8_t kReady{2};
constexpr uint8_t kBatch{3};
constexpr uint8_t kResult{4};
constexpr uint8_t kError{5};
constexpr uint8_t kEnd{6};

// A message header: the type and the payload length, little-endian.
constexpr size_t kHeaderSize{9};

// The longest message the worker may send; a longer length means a corrupt
// stream, such as user code writing to the socket.
constexpr uint64_t kMaxMessageBytes{2ULL << 30};

// The file descriptor the worker inherits its end of the socket on.
constexpr int kWorkerSocket{3};

ArrowOptions arrowOptions() {
  // The worker converts timestamps to the microsecond.
  return ArrowOptions{.timestampUnit = TimestampUnit::kMicro};
}

template <typename T>
T arrowValue(arrow::Result<T> result) {
  VELOX_CHECK(result.ok(), "Arrow failed: {}", result.status().ToString());
  return std::move(result).ValueUnsafe();
}

void checkArrow(const arrow::Status& status) {
  VELOX_CHECK(status.ok(), "Arrow failed: {}", status.ToString());
}

std::shared_ptr<arrow::Schema> toArrowSchema(
    const RowTypePtr& type,
    memory::MemoryPool* pool) {
  ArrowSchema schema;
  exportToArrow(BaseVector::create(type, 0, pool), schema, arrowOptions());
  return arrowValue(arrow::ImportSchema(&schema));
}

std::string toString(const arrow::Buffer& buffer) {
  return std::string{
      buffer.data_as<char>(), static_cast<size_t>(buffer.size())};
}

// Appends 'section' with its 4-byte little-endian length.
void appendSection(std::string& payload, std::string_view section) {
  VELOX_USER_CHECK_LE(
      section.size(),
      std::numeric_limits<uint32_t>::max(),
      "Python function setup section is too large");
  const auto length = static_cast<uint32_t>(section.size());
  payload.append(reinterpret_cast<const char*>(&length), sizeof(length));
  payload.append(section);
}

std::string makeSetupPayload(
    const PythonWorker::Options& options,
    memory::MemoryPool* pool) {
  folly::dynamic functions = folly::dynamic::array;
  std::vector<std::string> names;
  std::vector<TypePtr> resultTypes;
  for (const auto& function : options.functions) {
    folly::dynamic arguments = folly::dynamic::array;
    for (const auto& argument : function.arguments) {
      arguments.push_back(
          argument.column.has_value()
              ? folly::dynamic::object("column", argument.column.value())
              : folly::dynamic::object("function", argument.function.value()));
    }
    functions.push_back(
        folly::dynamic::object("name", function.name)(
            "arguments", std::move(arguments))("output", function.output));
    names.push_back(fmt::format("f{}", names.size()));
    resultTypes.push_back(function.resultType);
  }
  const folly::dynamic header =
      folly::dynamic::object("protocolVersion", kProtocolVersion)(
          "pythonVersion", options.pythonVersion)(
          // Converting with UTC leaves timestamps as they are.
          "sessionTimeZone",
          options.sessionTimeZone.value_or("UTC"))(
          "functions", std::move(functions));

  std::string payload;
  appendSection(payload, folly::toJson(header));
  const auto schema =
      toArrowSchema(ROW(std::move(names), std::move(resultTypes)), pool);
  appendSection(
      payload,
      toString(*arrowValue(
          arrow::ipc::SerializeSchema(*schema, arrow::default_memory_pool()))));
  for (const auto& function : options.functions) {
    appendSection(payload, function.pickledFunction);
  }
  return payload;
}

// Writes worker.py to a new temporary file that only this process could have
// created. Returns its path.
std::string writeWorkerScript() {
  std::string file =
      (std::filesystem::temp_directory_path() / "velox_python_worker_XXXXXX.py")
          .string();
  const int fd = ::mkstemps(file.data(), 3);
  VELOX_CHECK_GE(fd, 0, "Failed to create the Python worker script: {}", file);
  const auto script = workerScript();
  const bool written = folly::writeFull(fd, script.data(), script.size()) ==
      static_cast<ssize_t>(script.size());
  ::close(fd);
  VELOX_CHECK(written, "Failed to write the Python worker script: {}", file);
  return file;
}

// Returns true if 'vector' and its children, at any depth, are flat.
bool isFlat(const BaseVector& vector) {
  switch (vector.encoding()) {
    case VectorEncoding::Simple::FLAT:
      return true;
    case VectorEncoding::Simple::ROW:
      for (const auto& child : vector.asUnchecked<RowVector>()->children()) {
        if (!isFlat(*child)) {
          return false;
        }
      }
      return true;
    case VectorEncoding::Simple::ARRAY:
      return isFlat(*vector.asUnchecked<ArrayVector>()->elements());
    case VectorEncoding::Simple::MAP:
      return isFlat(*vector.asUnchecked<MapVector>()->mapKeys()) &&
          isFlat(*vector.asUnchecked<MapVector>()->mapValues());
    default:
      return false;
  }
}

// Reads the value of 'key' from a /proc/<pid>/status file, in bytes.
std::optional<uint64_t> statusBytes(
    std::string_view status,
    std::string_view key) {
  const auto start = status.find(key);
  if (start == std::string_view::npos) {
    return std::nullopt;
  }
  const auto kilobytes =
      std::strtoull(status.data() + start + key.size(), nullptr, 10);
  return kilobytes * 1'024;
}

// Reads the user and system CPU time from a /proc/<pid>/stat file.
std::optional<uint64_t> statCpuNanos(const std::string& stat) {
  // Fields after the parenthesized command name start at field 3; user and
  // system time are fields 14 and 15, in clock ticks.
  const auto end = stat.rfind(')');
  if (end == std::string::npos) {
    return std::nullopt;
  }
  std::vector<std::string_view> fields;
  folly::split(' ', std::string_view{stat}.substr(end + 2), fields);
  if (fields.size() < 13) {
    return std::nullopt;
  }
  const auto ticks =
      folly::to<uint64_t>(fields[11]) + folly::to<uint64_t>(fields[12]);
  return ticks * 1'000'000'000 / ::sysconf(_SC_CLK_TCK);
}

// Returns the process's environment with 'additions' added. An addition
// replaces an inherited variable of the same name.
std::vector<std::string> processEnvironment(
    const std::vector<std::string>& additions) {
  const auto name = [](std::string_view variable) {
    return variable.substr(0, variable.find('='));
  };
  folly::F14FastSet<std::string_view> addedNames;
  for (const auto& addition : additions) {
    addedNames.insert(name(addition));
  }
  std::vector<std::string> environment;
  for (auto** variable = environ; *variable != nullptr; ++variable) {
    if (!addedNames.contains(name(*variable))) {
      environment.emplace_back(*variable);
    }
  }
  environment.insert(environment.end(), additions.begin(), additions.end());
  return environment;
}

// Returns a user error: the functions or their environment are at fault.
std::exception_ptr userError(std::string_view error) {
  try {
    VELOX_USER_FAIL("Python worker failed: {}", error);
  } catch (const VeloxUserError&) {
    return std::current_exception();
  }
}

// Returns a runtime error: the worker or the system failed.
std::exception_ptr runtimeError(std::string_view error) {
  try {
    VELOX_FAIL("Python worker failed: {}", error);
  } catch (const VeloxRuntimeError&) {
    return std::current_exception();
  }
}

// Returns the exception being handled as a user or runtime error. Call only
// from a catch block.
std::exception_ptr currentError() {
  try {
    throw;
  } catch (const VeloxUserError& e) {
    return userError(e.message());
  } catch (const VeloxException& e) {
    return runtimeError(e.message());
  } catch (const std::exception& e) {
    return runtimeError(e.what());
  }
}

// Returns the error an ERROR message reports: a user error with the message
// and traceback of what the functions raised.
std::exception_ptr workerError(std::string_view payload) {
  try {
    const auto error = folly::parseJson(payload);
    return userError(
        fmt::format(
            "{}\n{}",
            error["message"].asString(),
            error["traceback"].asString()));
  } catch (const std::exception&) {
    return runtimeError(
        fmt::format("Python worker sent a malformed ERROR: {}", payload));
  }
}

} // namespace

PythonWorker::PythonWorker(Options options, memory::MemoryPool* pool)
    : pool_{pool},
      environmentUri_{std::move(options.environmentUri)},
      resolvers_{std::move(options.resolvers)},
      setupPayload_{makeSetupPayload(options, pool)},
      writer_{[this] { write(); }} {
  // A thread left running when the constructor throws would end the process.
  try {
    reader_ = std::thread([this] { read(); });
  } catch (...) {
    stop(/*waitForEnd=*/false);
    throw;
  }
}

PythonWorker::~PythonWorker() {
  stop(/*waitForEnd=*/false);
}

void PythonWorker::stop(bool waitForEnd) {
  {
    std::unique_lock<std::mutex> lock{mutex_};
    if (stopped_) {
      return;
    }
    stopped_ = true;
    // A finished worker answers END and exits on its own.
    if (waitForEnd && finishing_) {
      stateChanged_.wait_for(lock, std::chrono::seconds(10), [&] {
        return finalStats_.has_value() || error_ != nullptr;
      });
    }
    stopping_ = true;
    if (socket_ >= 0) {
      ::shutdown(socket_, SHUT_RDWR);
    }
  }
  stateChanged_.notify_all();
  writer_.join();
  if (reader_.joinable()) {
    reader_.join();
  }
  // Reads the stats while the process is still this worker's: once reaped,
  // its pid may belong to another process.
  processStats();
  if (process_.has_value() && process_->poll().running()) {
    process_->kill();
    process_->wait();
  }
  pid_ = 0;
  if (!scriptPath_.empty()) {
    std::filesystem::remove(scriptPath_);
  }
  if (socket_ >= 0) {
    ::close(socket_);
  }
  // A driver may still wait on receive().
  std::unique_lock<std::mutex> lock{mutex_};
  fulfillPromises(lock);
}

void PythonWorker::send(const RowVectorPtr& arguments) {
  auto columns = arguments->children();
  for (auto& column : columns) {
    column = BaseVector::loadedVectorShared(column);
    // The input passes through to the output, so it is copied, not flattened
    // in place. A constant goes to Arrow as a single run. A complex value
    // that is encoded is copied flat first: one row, not one per row of the
    // batch.
    if (column->isConstantEncoding()) {
      if (!column->type()->isPrimitiveType() &&
          !isFlat(*column->valueVector())) {
        column = BaseVector::wrapInConstant(
            column->size(), 0, BaseVector::copy(*column->slice(0, 1), pool_));
      }
    } else if (!isFlat(*column)) {
      column = BaseVector::copy(*column, pool_);
    }
  }
  const auto flat = std::make_shared<RowVector>(
      pool_, arguments->type(), nullptr, arguments->size(), std::move(columns));

  ArrowArray array;
  ArrowSchema schema;
  exportToArrow(flat, array, pool_, arrowOptions());
  exportToArrow(flat, schema, arrowOptions());
  const auto batch = arrowValue(arrow::ImportRecordBatch(&array, &schema));
  const auto stream = arrowValue(arrow::io::BufferOutputStream::Create());
  const auto writer =
      arrowValue(arrow::ipc::MakeStreamWriter(stream, batch->schema()));
  checkArrow(writer->WriteRecordBatch(*batch));
  checkArrow(writer->Close());
  auto payload = toString(*arrowValue(stream->Finish()));

  {
    std::lock_guard<std::mutex> lock{mutex_};
    outgoing_.push_back(std::move(payload));
  }
  stateChanged_.notify_all();
}

std::optional<PythonWorker::Result> PythonWorker::receive(
    ContinueFuture* future) {
  std::string payload;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    if (incoming_.empty()) {
      if (error_ != nullptr) {
        std::rethrow_exception(error_);
      }
      if (future != nullptr) {
        auto [promise, promiseFuture] =
            makeVeloxContinuePromiseContract("PythonWorker::receive");
        promises_.push_back(std::move(promise));
        *future = std::move(promiseFuture);
      }
      return std::nullopt;
    }
    payload = std::move(incoming_.front());
    incoming_.pop_front();
  }

  const auto reader = arrowValue(
      arrow::ipc::RecordBatchStreamReader::Open(
          std::make_shared<arrow::io::BufferReader>(
              arrow::Buffer::FromString(std::move(payload)))));
  std::shared_ptr<arrow::RecordBatch> batch;
  checkArrow(reader->ReadNext(&batch));
  VELOX_CHECK_NOT_NULL(batch, "Python worker sent a RESULT without a batch");
  uint64_t functionWallNanos{0};
  if (const auto& metadata = batch->schema()->metadata()) {
    const auto seconds = metadata->Get("functionWallSeconds");
    if (seconds.ok()) {
      for (const auto& value : folly::parseJson(seconds.ValueUnsafe())) {
        functionWallNanos += static_cast<uint64_t>(value.asDouble() * 1e9);
      }
    }
  }
  ArrowArray array;
  ArrowSchema schema;
  checkArrow(arrow::ExportRecordBatch(*batch, &array, &schema));
  const auto imported = importFromArrowAsOwner(schema, array, pool_);
  // A copy in the pool, so the memory counts against the query.
  return Result{
      .columns = std::dynamic_pointer_cast<RowVector>(
          BaseVector::copy(*imported, pool_)),
      .functionWallNanos = functionWallNanos,
  };
}

uint64_t PythonWorker::bufferedBytes() const {
  std::lock_guard<std::mutex> lock{mutex_};
  uint64_t bytes{0};
  for (const auto& payload : outgoing_) {
    bytes += payload.size();
  }
  for (const auto& payload : incoming_) {
    bytes += payload.size();
  }
  return bytes;
}

std::optional<uint64_t> PythonWorker::unpickleNanos() const {
  std::lock_guard<std::mutex> lock{mutex_};
  return unpickleNanos_;
}

PythonWorker::ProcessStats PythonWorker::processStats() {
  {
    std::lock_guard<std::mutex> lock{mutex_};
    if (finalStats_.has_value()) {
      return finalStats_.value();
    }
  }
  const auto pid = pid_.load();
  if (pid == 0) {
    return lastStats_;
  }
  std::string stat;
  if (folly::readFile(fmt::format("/proc/{}/stat", pid).c_str(), stat)) {
    if (const auto cpuNanos = statCpuNanos(stat)) {
      lastStats_.cpuNanos = cpuNanos.value();
    }
  }
  std::string status;
  if (folly::readFile(fmt::format("/proc/{}/status", pid).c_str(), status)) {
    if (const auto peak = statusBytes(status, "VmHWM:")) {
      lastStats_.peakMemoryBytes = peak.value();
    }
  }
  return lastStats_;
}

void PythonWorker::finish() {
  {
    std::lock_guard<std::mutex> lock{mutex_};
    finishing_ = true;
  }
  stateChanged_.notify_all();
}

void PythonWorker::startProcess() {
  const auto interpreter = resolvers_->resolve(environmentUri_);
  std::vector<std::string> argv{interpreter.path};
  argv.insert(
      argv.end(), interpreter.arguments.begin(), interpreter.arguments.end());
  scriptPath_ = writeWorkerScript();
  argv.push_back(scriptPath_);
  argv.push_back("--fd");
  argv.push_back(std::to_string(kWorkerSocket));
  const auto environment = processEnvironment(interpreter.environmentVariables);

  int sockets[2];
  VELOX_CHECK_EQ(
      ::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets),
      0,
      "Failed to create a socket pair for the Python worker");
  try {
    process_.emplace(
        argv,
        folly::Subprocess::Options()
            .fd(kWorkerSocket, sockets[1])
            .closeOtherFds()
            .parentDeathSignal(SIGKILL),
        nullptr,
        &environment);
  } catch (...) {
    ::close(sockets[0]);
    ::close(sockets[1]);
    throw;
  }
  ::close(sockets[1]);
  pid_ = process_->pid();

  {
    std::lock_guard<std::mutex> lock{mutex_};
    socket_ = sockets[0];
    if (stopping_) {
      ::shutdown(socket_, SHUT_RDWR);
    }
  }
  stateChanged_.notify_all();
}

namespace {

// Writes all of 'data'. MSG_NOSIGNAL keeps a closed socket from raising
// SIGPIPE, which would end the process.
bool sendAll(int socket, const char* data, size_t size) {
  while (size > 0) {
    const auto sent = ::send(socket, data, size, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    data += sent;
    size -= sent;
  }
  return true;
}

} // namespace

bool PythonWorker::writeMessage(uint8_t type, std::string_view payload) const {
  char header[kHeaderSize];
  header[0] = static_cast<char>(type);
  const uint64_t length = payload.size();
  std::memcpy(header + 1, &length, sizeof(length));
  return sendAll(socket_, header, kHeaderSize) &&
      sendAll(socket_, payload.data(), payload.size());
}

std::optional<std::pair<uint8_t, std::string>> PythonWorker::readMessage()
    const {
  char header[kHeaderSize];
  if (folly::readFull(socket_, header, kHeaderSize) != kHeaderSize) {
    return std::nullopt;
  }
  uint64_t length;
  std::memcpy(&length, header + 1, sizeof(length));
  VELOX_CHECK_LE(
      length,
      kMaxMessageBytes,
      "Python worker sent a message that is too long");
  std::string payload(length, '\0');
  if (folly::readFull(socket_, payload.data(), length) !=
      static_cast<ssize_t>(length)) {
    return std::nullopt;
  }
  return std::make_pair(static_cast<uint8_t>(header[0]), std::move(payload));
}

void PythonWorker::write() {
  try {
    startProcess();
  } catch (const std::exception&) {
    fail(currentError());
    return;
  }
  // From here on the reader reports failures: a write fails when the worker
  // exited, and the reader then reads its ERROR or the end of the stream.
  if (!writeMessage(kSetup, setupPayload_)) {
    return;
  }
  std::unique_lock<std::mutex> lock{mutex_};
  while (true) {
    stateChanged_.wait(lock, [&] {
      return stopping_ || error_ != nullptr || finishing_ || !outgoing_.empty();
    });
    if (stopping_ || error_ != nullptr) {
      break;
    }
    if (outgoing_.empty()) {
      lock.unlock();
      writeMessage(kEnd, "");
      lock.lock();
      break;
    }
    auto payload = std::move(outgoing_.front());
    outgoing_.pop_front();
    lock.unlock();
    if (!writeMessage(kBatch, payload)) {
      lock.lock();
      break;
    }
    lock.lock();
    ++numBatchesSent_;
  }
  // The worker dies with the thread that started it (parentDeathSignal
  // follows the forking thread), so this thread outlives the worker.
  stateChanged_.wait(lock, [&] { return stopping_; });
}

void PythonWorker::read() {
  // An exception on this thread would end the process.
  try {
    readMessages();
  } catch (const std::exception&) {
    fail(currentError());
  }
}

void PythonWorker::readMessages() {
  {
    std::unique_lock<std::mutex> lock{mutex_};
    stateChanged_.wait(
        lock, [&] { return stopping_ || error_ != nullptr || socket_ >= 0; });
    if (socket_ < 0) {
      return;
    }
  }
  auto message = readMessage();
  if (!message.has_value()) {
    fail(runtimeError(
        "Python worker exited before it was ready; its stderr may say why"));
    return;
  }
  if (message->first == kError) {
    fail(workerError(message->second));
    return;
  }
  if (message->first != kReady) {
    fail(runtimeError(
        fmt::format(
            "Python worker must answer SETUP with READY: {}", message->first)));
    return;
  }
  try {
    const auto seconds =
        folly::parseJson(message->second)["unpickleSeconds"].asDouble();
    std::lock_guard<std::mutex> lock{mutex_};
    unpickleNanos_ = static_cast<uint64_t>(seconds * 1e9);
  } catch (const std::exception&) {
    fail(runtimeError(
        fmt::format(
            "Python worker sent a malformed READY: {}", message->second)));
    return;
  }
  while ((message = readMessage()).has_value()) {
    if (message->first == kEnd) {
      bool resultsMissing{false};
      {
        std::lock_guard<std::mutex> lock{mutex_};
        resultsMissing = numResults_ < numBatchesSent_;
      }
      if (resultsMissing) {
        fail(runtimeError("Python worker must answer every BATCH before END"));
        return;
      }
      try {
        const auto stats = folly::parseJson(message->second);
        std::lock_guard<std::mutex> lock{mutex_};
        finalStats_ = ProcessStats{
            .cpuNanos =
                static_cast<uint64_t>(stats["cpuSeconds"].asDouble() * 1e9),
            .peakMemoryBytes =
                static_cast<uint64_t>(stats["peakMemoryBytes"].asInt()),
        };
      } catch (const std::exception&) {
        fail(runtimeError(
            fmt::format(
                "Python worker sent a malformed END: {}", message->second)));
        return;
      }
      stateChanged_.notify_all();
      return;
    }
    if (message->first == kError) {
      fail(workerError(message->second));
      return;
    }
    if (message->first != kResult) {
      fail(runtimeError(
          fmt::format(
              "Python worker must answer BATCH with RESULT: {}",
              message->first)));
      return;
    }
    std::unique_lock<std::mutex> lock{mutex_};
    incoming_.push_back(std::move(message->second));
    ++numResults_;
    fulfillPromises(lock);
  }
  fail(runtimeError(
      "Python worker exited without reporting an error; its stderr may say why"));
}

void PythonWorker::fail(std::exception_ptr error) {
  std::unique_lock<std::mutex> lock{mutex_};
  if (stopping_ || error_ != nullptr) {
    return;
  }
  error_ = std::move(error);
  stateChanged_.notify_all();
  fulfillPromises(lock);
}

void PythonWorker::fulfillPromises(std::unique_lock<std::mutex>& lock) {
  auto promises = std::move(promises_);
  promises_.clear();
  lock.unlock();
  for (auto& promise : promises) {
    promise.setValue();
  }
}

} // namespace facebook::velox::exec::python
