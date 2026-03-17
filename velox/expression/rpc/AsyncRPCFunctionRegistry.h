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

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "velox/common/base/Exceptions.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// Registry for AsyncRPCFunction implementations.
///
/// This registry allows RPC functions to register themselves by name,
/// enabling RPCOperator to resolve the function at operator init time
/// (following the VectorFunction pattern where plan nodes reference
/// functions by name and the execution layer resolves them).
///
/// Usage (in function's .cpp file):
///   AsyncRPCFunctionRegistry::registerFunction(
///       "my_function",
///       []() { return std::make_shared<MyAsyncRPCFunction>(); });
///
/// Resolution (in RPCOperator::initialize()):
///   auto func = AsyncRPCFunctionRegistry::create("my_function");
class AsyncRPCFunctionRegistry {
 public:
  /// Factory function type that creates an AsyncRPCFunction instance.
  using Factory = std::function<std::shared_ptr<AsyncRPCFunction>()>;

  /// Registers an AsyncRPCFunction factory for the given function name.
  /// Thread-safe. Safe to call during static initialization.
  ///
  /// @param name Function name (e.g., "my_rpc_function")
  /// @param factory Function that creates instances of the AsyncRPCFunction
  /// @return true if registration succeeded, false if name already registered
  static bool registerFunction(const std::string& name, Factory factory);

  /// Creates an AsyncRPCFunction instance for the given function name.
  /// Thread-safe.
  ///
  /// @param name Function name to look up
  /// @return AsyncRPCFunction instance, or nullptr if not registered
  static std::shared_ptr<AsyncRPCFunction> create(const std::string& name);

  /// Checks if a function name is registered.
  /// Thread-safe.
  ///
  /// @param name Function name to check
  /// @return true if the function is registered
  static bool isRegistered(const std::string& name);

  /// Returns all registered function names.
  /// Thread-safe.
  ///
  /// @return Set of registered function names
  static std::unordered_set<std::string> registeredFunctions();

  /// Clears all registered functions.
  /// Primarily for testing.
  static void clear();

 private:
  static std::mutex& mutex();
  static std::unordered_map<std::string, Factory>& factories();
};

/// Helper class for static registration of AsyncRPCFunction implementations.
class AsyncRPCFunctionRegistrar {
 public:
  AsyncRPCFunctionRegistrar(
      const std::string& name,
      AsyncRPCFunctionRegistry::Factory factory) {
    AsyncRPCFunctionRegistry::registerFunction(name, std::move(factory));
  }
};

} // namespace facebook::velox::exec::rpc
