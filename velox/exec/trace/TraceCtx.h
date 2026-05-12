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
#include <string_view>
#include "velox/exec/trace/TraceWriter.h"

namespace facebook::velox::exec {
class Operator;
class VectorFunction;
} // namespace facebook::velox::exec

namespace facebook::velox::exec::trace {

class TraceCtx {
 public:
  TraceCtx(bool dryRun) : dryRun_(dryRun) {}

  virtual ~TraceCtx() = default;

  /// Overwrite the methods below to provide a concrete trace writer
  /// implementation for input, split, and task metadata.
  virtual std::unique_ptr<trace::TraceInputWriter> createInputTracer(
      Operator&) const {
    return nullptr;
  }

  virtual std::unique_ptr<trace::TraceSplitWriter> createSplitTracer(
      Operator&) const {
    return nullptr;
  }

  virtual std::unique_ptr<trace::TraceMetadataWriter> createMetadataTracer()
      const {
    return nullptr;
  }

  /// Returns whether a particular operator should be traced. Called before the
  /// task starts execution, when operators are instantiated.
  virtual bool shouldTrace(const Operator&) const {
    return false;
  }

  /// Returns whether a particular expression function should be traced. Called
  /// during operator initialization in Expr::maybeSetupTracer().
  virtual bool shouldTraceExpr(std::string_view /*functionName*/) const {
    return false;
  }

  /// Creates a writer for capturing output batches from a traced expression.
  /// The instanceIndex distinguishes multiple Expr nodes with the same
  /// function name within a single operator. Ownership is transferred to the
  /// caller.
  virtual std::unique_ptr<trace::TraceExprWriter> createExprOutputTracer(
      const Operator& /*op*/,
      std::string_view /*functionName*/,
      int /*instanceIndex*/) const {
    return nullptr;
  }

  /// Creates a writer for capturing input batches to a traced expression.
  virtual std::unique_ptr<trace::TraceExprInputWriter> createExprInputTracer(
      const Operator& /*op*/,
      std::string_view /*functionName*/,
      int /*instanceIndex*/) const {
    return nullptr;
  }

  /// Activates intra-expression tracing on a vector function whose owning
  /// operator and expression have both passed the tracing gates
  /// (shouldTrace() and shouldTraceExpr() respectively).
  virtual void maybeActivateIntraExprTracing(
      const Operator& /*op*/,
      std::string_view /*functionName*/,
      VectorFunction& /*function*/) const {}

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
