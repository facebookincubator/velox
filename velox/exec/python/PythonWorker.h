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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include <folly/Subprocess.h>

#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/python/PythonEnvironmentResolver.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::python {

/// An argument of a Python function. Exactly one member is set.
struct PythonArgument {
  /// A column of the batches sent, by index.
  std::optional<column_index_t> column;

  /// The results of an earlier function, by its index among the functions.
  std::optional<size_t> function;
};

/// A function for a worker to run, as the SETUP message describes it.
struct PythonFunction {
  /// The function's name, for error messages.
  std::string name;

  /// A pickle of a callable, or of a tuple whose first item is the callable.
  std::string pickledFunction;

  /// The type the function's results must have.
  TypePtr resultType;

  /// The function's arguments, in order.
  std::vector<PythonArgument> arguments;

  /// Whether results come back to Velox; otherwise only later functions read
  /// them.
  bool output{true};
};

/// Runs Python functions in a worker process, which speaks the protocol in
/// velox/exec/python/README.md. Two background threads start the process and
/// move messages over its socket, one writing and one reading, so that large
/// batches and results never block each other. Arrow conversion and memory
/// allocation run on the caller's thread.
class PythonWorker {
 public:
  struct Options {
    /// URI of the Python environment, resolved by 'resolvers'.
    std::string environmentUri;

    /// The resolvers of the query's environment URIs.
    std::shared_ptr<const PythonEnvironmentResolvers> resolvers;

    /// The Python major.minor version the functions were pickled with.
    std::string pythonVersion;

    /// The time zone timestamps are converted with: an IANA name or a fixed
    /// offset such as +05:45. Timestamps are used as they are if not set.
    std::optional<std::string> sessionTimeZone;

    /// The functions, each after the functions it reads.
    std::vector<PythonFunction> functions;
  };

  /// The results for one batch.
  struct Result {
    /// Per output function, in order, a result column and a VARCHAR error
    /// column.
    RowVectorPtr columns;

    /// Wall time the worker spent in the functions on the batch.
    uint64_t functionWallNanos{0};
  };

  /// CPU time and peak resident memory of the worker process.
  struct ProcessStats {
    /// User and system CPU time.
    uint64_t cpuNanos{0};

    /// Peak resident memory.
    uint64_t peakMemoryBytes{0};
  };

  /// Starts the worker in the background and returns without waiting.
  PythonWorker(Options options, memory::MemoryPool* pool);

  /// Calls stop() without waiting for END.
  ~PythonWorker();

  /// Stops the worker and its threads, killing the process if it is still
  /// running. With 'waitForEnd', after finish(), first waits a few seconds for
  /// the worker to answer END, with its final stats, and exit. Idempotent.
  /// Waits for the environment's resolver if it is still running, which
  /// cannot be cancelled.
  void stop(bool waitForEnd);

  /// Converts a batch of argument columns to Arrow and queues it to send.
  /// Never waits for the worker.
  void send(const RowVectorPtr& arguments);

  /// Returns the results for the oldest batch sent and not yet received,
  /// allocated in the pool. Returns std::nullopt if they have not arrived yet,
  /// and sets 'future' to complete when they do unless 'future' is null.
  /// Throws if the worker failed.
  std::optional<Result> receive(ContinueFuture* future);

  /// Asks the worker to exit after the queued batches. Never blocks.
  void finish();

  /// Returns the time the worker took to unpickle the functions and the
  /// modules they import, once it is ready.
  std::optional<uint64_t> unpickleNanos() const;

  /// Returns the worker process's CPU time and peak resident memory. Once the
  /// worker answered END, these are the final values it reported. Otherwise
  /// they are read from /proc while the process is this worker's; after stop()
  /// they are the last values read.
  ProcessStats processStats();

  /// Returns the bytes of the batches and results held in memory: queued to
  /// send, or received and not yet returned by receive().
  uint64_t bufferedBytes() const;

 private:
  // Starts the process and sends SETUP, then sends batches until END.
  void write();

  // Runs readMessages() and records any exception as the worker's failure.
  void read();

  // Reads READY, then RESULT messages until END.
  void readMessages();

  // Resolves the environment and starts the process over a socket pair.
  void startProcess();

  // Writes one message. Returns false if the socket failed, as when the worker
  // exited.
  bool writeMessage(uint8_t type, std::string_view payload) const;

  // Reads one message. Returns std::nullopt if the socket closed first.
  std::optional<std::pair<uint8_t, std::string>> readMessage() const;

  // Records the worker's failure, which the next receive() throws.
  void fail(std::string message);

  // Fulfills the promises receive() handed out. Unlocks 'lock'.
  void fulfillPromises(std::unique_lock<std::mutex>& lock);

  memory::MemoryPool* const pool_;
  const std::string environmentUri_;
  const std::shared_ptr<const PythonEnvironmentResolvers> resolvers_;
  const std::string setupPayload_;

  // Guards the members below it, up to 'socket_'.
  mutable std::mutex mutex_;
  std::condition_variable stateChanged_;
  // Payloads of BATCH messages to send, oldest first.
  std::deque<std::string> outgoing_;
  // Payloads of RESULT messages received, oldest first.
  std::deque<std::string> incoming_;
  std::vector<ContinuePromise> promises_;
  // Set by finish(): send END after the queued batches.
  bool finishing_{false};
  // Set by stop(): the threads exit.
  bool stopping_{false};
  // Set once the worker failed: the error the next receive() throws.
  std::optional<std::string> error_;
  std::optional<uint64_t> unpickleNanos_;
  // The final stats the worker reported in END.
  std::optional<ProcessStats> finalStats_;
  // Set by the first stop().
  bool stopped_{false};
  // The executor's end of the socket pair; -1 until the process starts.
  int socket_{-1};

  // The process, once started. Read by processStats() on the caller's thread.
  std::atomic<pid_t> pid_{0};
  // The stats last read from /proc, on the caller's thread.
  ProcessStats lastStats_;
  // Set on the writer thread; read after it is joined.
  std::optional<folly::Subprocess> process_;
  std::thread writer_;
  std::thread reader_;
};

} // namespace facebook::velox::exec::python
