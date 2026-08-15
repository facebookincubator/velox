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
#include <vector>

#include "velox/expression/FunctionSignature.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// Registry for AsyncRPCFunction implementations.
///
/// == Why a separate registry (not the VectorFunction registry)? ==
///
/// AsyncRPCFunctions execute via RPCOperator, completely outside the
/// expression evaluation framework. The Velox VectorFunction registry
/// (vectorFunctionFactories()) is designed for synchronous, expression-level
/// functions resolved by ExprCompiler with SignatureBinder type matching.
/// RPC functions differ in several ways:
///
///   1. **Execution model**: VectorFunctions run synchronously inside
///      Expr::eval(). AsyncRPCFunctions run asynchronously via RPCOperator
///      with futures, rate limiting, and backpressure — not part of the
///      expression tree at all.
///
///   2. **No type resolution needed**: VectorFunctions need SignatureBinder
///      because the same name can have multiple overloads (e.g., concat for
///      varchar vs array). RPC functions have fixed signatures — the Java
///      planner already resolved types before creating the RPCNode.
///
///   3. **Factory signature mismatch**: VectorFunctionFactory takes
///      (name, inputArgs, config) and returns shared_ptr<VectorFunction>.
///      Our factory returns shared_ptr<AsyncRPCFunction> with no args;
///      initialization happens separately via initialize().
///
///   4. **Stub bridge for discovery**: We register lightweight stubs in the
///      VectorFunction registry purely for sidecar /v1/functions discovery.
///      The stubs throw on direct execution — actual execution goes through
///      RPCNode → RPCOperator → AsyncRPCFunction.
///
/// Usage (in function's .cpp file):
///   // For a complete example, see velox/exec/rpc/tests/DemoRPCFunction*.
///
///   #include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"
///   VELOX_REGISTER_RPC_FUNCTION(my_function, MyAsyncRPCFunction);
///
/// Lookup (in RPCOperator::initialize()):
///   auto func = AsyncRPCFunctionRegistry::create("my_function");
class AsyncRPCFunctionRegistry {
 public:
  /// Factory function type that creates an AsyncRPCFunction instance.
  using Factory = std::function<std::shared_ptr<AsyncRPCFunction>()>;

  /// Signature list type for stub registration.
  using Signatures = std::vector<std::shared_ptr<exec::FunctionSignature>>;

  /// Registers an AsyncRPCFunction factory for the given function name.
  /// Thread-safe. Safe to call during static initialization.
  ///
  /// @param name Function name (e.g., "my_rpc_function")
  /// @param factory Function that creates instances of the AsyncRPCFunction
  /// @return true if registration succeeded, false if name already registered
  static bool registerFunction(const std::string& name, Factory factory);

  /// Registers a function factory AND its signatures for stub registration.
  /// The factory is registered immediately. The signatures are stored and
  /// registered as stub Velox functions later via registerStubs().
  /// Thread-safe. Safe to call during static initialization.
  static bool registerFunction(
      const std::string& name,
      Factory factory,
      Signatures signatures);

  /// Registers stub Velox functions for all functions that provided signatures.
  /// Must be called during server startup (after config is available).
  ///
  /// For each registered function with signatures, registers a stub using
  /// the given namespace prefix: namespacePrefix + functionName
  /// e.g., "presto.default.fb_llm_inference"
  ///
  /// @param namespacePrefix The catalog.schema with trailing dot (e.g.,
  /// "presto.default.")
  static void registerStubs(const std::string& namespacePrefix);

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
  /// Intended ONLY for unit tests to avoid test contamination.
  /// WARNING: Do NOT call this in production code.
  static void testingClear();

 private:
  static std::mutex& mutex();
  static std::unordered_map<std::string, Factory>& factories();
  static std::unordered_map<std::string, Signatures>& signatureStore();

  /// Registers an entry under the lock. Caller must hold mutex().
  static bool registerEntryLocked(
      const std::string& name,
      Factory factory,
      Signatures signatures);
};

/// Helper class for static registration of AsyncRPCFunction implementations.
///
/// Two-argument form: registers the factory only (for RPCOperator lookup).
/// Three-argument form: also stores signatures for deferred stub registration
/// via registerStubs(), enabling automatic sidecar discovery.
class AsyncRPCFunctionRegistrar {
 public:
  AsyncRPCFunctionRegistrar(
      const std::string& name,
      AsyncRPCFunctionRegistry::Factory factory) {
    AsyncRPCFunctionRegistry::registerFunction(name, std::move(factory));
  }

  AsyncRPCFunctionRegistrar(
      const std::string& name,
      AsyncRPCFunctionRegistry::Factory factory,
      AsyncRPCFunctionRegistry::Signatures signatures) {
    AsyncRPCFunctionRegistry::registerFunction(
        name, std::move(factory), std::move(signatures));
  }
};

/// Convenience macros for registering AsyncRPCFunction implementations.
/// Place in the function's .cpp file (at namespace scope).
///
/// The ClassName must provide:
///   - A default constructor (or be constructible via std::make_shared)
///   - A static signatures() method returning
///     std::vector<std::shared_ptr<exec::FunctionSignature>>
///
/// BUCK target must set link_whole = True to prevent linker stripping.

// Internal helper — generates a unique variable name per line.
#define _VELOX_RPC_REGISTRAR_VAR(name, line) __velox_rpc_reg_##name##_##line
#define _VELOX_RPC_REGISTRAR_VAR2(name, line) \
  _VELOX_RPC_REGISTRAR_VAR(name, line)

/// Register an RPC function with default make_shared factory + signatures().
/// Usage: VELOX_REGISTER_RPC_FUNCTION(my_rpc, MyAsyncRPCFunction);
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
#define VELOX_REGISTER_RPC_FUNCTION(name, ClassName)                           \
  static ::facebook::velox::exec::rpc::AsyncRPCFunctionRegistrar               \
  _VELOX_RPC_REGISTRAR_VAR2(name, __LINE__)(                                   \
      #name,                                                                   \
      []()                                                                     \
          -> std::shared_ptr<::facebook::velox::exec::rpc::AsyncRPCFunction> { \
            return std::make_shared<ClassName>();                              \
          },                                                                   \
      ClassName::signatures())

/// Register an RPC function with a custom factory and explicit signatures.
/// Usage: VELOX_REGISTER_RPC_FUNCTION_CUSTOM_FACTORY(
///            my_rpc, myFactoryFn, MyClass::signatures());
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
#define VELOX_REGISTER_RPC_FUNCTION_CUSTOM_FACTORY(name, factory, sigs) \
  static ::facebook::velox::exec::rpc::AsyncRPCFunctionRegistrar        \
  _VELOX_RPC_REGISTRAR_VAR2(name, __LINE__)(#name, (factory), (sigs))

} // namespace facebook::velox::exec::rpc
