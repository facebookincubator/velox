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

#include <cstdint>
#include <functional>
#include <string>
#include "velox/exec/trace/TraceWriter.h"

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::exec::trace {

class TraceCtx {
 public:
  TraceCtx(bool dryRun) : dryRun_(dryRun) {}

  virtual ~TraceCtx() = default;

  virtual std::unique_ptr<trace::TraceInputWriter> createInputTracer(
      Operator& op) const = 0;

  virtual std::unique_ptr<trace::TraceSplitWriter> createSplitTracer(
      Operator& op) const = 0;

  virtual std::unique_ptr<trace::TraceMetadataWriter> createMetadataTracer()
      const = 0;

  virtual bool shouldTrace(const Operator& op) const = 0;

  bool dryRun() const {
    return dryRun_;
  }

 private:
  /// If true, we only collect operator input trace without the actual
  /// execution. This is used by crash debugging so that we can collect the
  /// input that triggers the crash.
  bool dryRun_{false};
};

} // namespace facebook::velox::exec::trace
