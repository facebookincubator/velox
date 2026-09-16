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
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/expression/FunctionMetadata.h"
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
///   2. **Resolution happens in the planner**: a planner resolves a call
///      against the signatures registered here, via find(), and records the
///      result type in the plan. RPCOperator executes an already-resolved
///      call, so neither ExprCompiler nor SignatureBinder is involved at
///      execution time.
///
///   3. **Factory signature mismatch**: VectorFunctionFactory takes
///      (name, inputArgs, config) and returns shared_ptr<VectorFunction>.
///      Our factory returns shared_ptr<AsyncRPCFunction> with no args;
///      initialization happens separately via initialize().
///
///   4. **Stub bridge for discovery**: registerStubs() registers a throwing
///      VectorFunction per name so the sidecar's /v1/functions endpoint, which
///      reads the VectorFunction registry, lists them. Deprecated: publish
///      from functions() instead.
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

  /// Declaration attached to the stub. Read by the sidecar, not by any
  /// executor: RPC functions run in RPCOperator and never reach the stub.
  using Metadata = exec::VectorFunctionMetadata;

  /// Registers a function factory and the signatures it accepts. Signatures
  /// must not be empty: a function nothing can resolve a call to is not
  /// registrable. The function declares the default metadata, which asserts
  /// determinism and that a NULL in any argument means a NULL result.
  /// Thread-safe. Safe to call during static initialization.
  ///
  /// @param name Function name (e.g., "my_rpc_function")
  /// @param factory Function that creates instances of the AsyncRPCFunction
  /// @return true if registration succeeded, false if name already registered
  static bool registerFunction(
      const std::string& name,
      Factory factory,
      Signatures signatures);

  /// Registers a factory, its signatures, and the metadata the stub is declared
  /// with. Use this when the default declaration is wrong for the function --
  /// most importantly when a NULL in one argument does not mean a NULL result,
  /// which the default 'defaultNullBehavior{true}' asserts.
  static bool registerFunction(
      const std::string& name,
      Factory factory,
      Signatures signatures,
      Metadata metadata);

  /// What a registered function declares: enough to resolve a call to it, or
  /// to publish it, without creating an instance.
  struct FunctionEntry {
    /// Name the function is registered under, without a namespace prefix.
    std::string name;

    /// Signatures the function accepts. Never empty.
    Signatures signatures;

    /// Determinism and null behavior.
    Metadata metadata;
  };

  /// Returns what 'name' declares, or nullopt when it is not registered.
  /// Thread-safe.
  static std::optional<FunctionEntry> find(const std::string& name);

  /// Returns what every registered function declares, in unspecified order.
  /// Thread-safe.
  static std::vector<FunctionEntry> functions();

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

  /// Creates an AsyncRPCFunction instance for the given function name.
  /// Thread-safe.
  ///
  /// @param name Function name to look up
  /// @return AsyncRPCFunction instance, or nullptr if not registered
  static std::shared_ptr<AsyncRPCFunction> create(const std::string& name);

  /// Deprecated. Use functions() and publish from it: a stub is a throwing
  /// VectorFunction in the scalar registry, which exists only because
  /// /v1/functions reads that registry. Removed once the sidecar publishes
  /// from functions() instead.
  ///
  /// Registers a stub for each registered function under the given namespace
  /// prefix: namespacePrefix + functionName, e.g.
  /// "presto.default.fb_llm_inference". Must be called during server startup,
  /// after config is available.
  ///
  /// @param namespacePrefix The catalog.schema with trailing dot (e.g.,
  /// "presto.default.")
  static void registerStubs(const std::string& namespacePrefix);

  /// Clears all registered functions.
  /// Intended ONLY for unit tests to avoid test contamination.
  /// WARNING: Do NOT call this in production code.
  static void testingClear();

 private:
  // What is stored per registered name: how to make one, and what it declares.
  struct Registration {
    Factory factory;
    Signatures signatures;
    Metadata metadata;
  };

  static std::mutex& mutex();
  static std::unordered_map<std::string, Registration>& registrations();
};

/// Helper class for static registration of AsyncRPCFunction implementations.
///
/// Three-argument form registers the factory and its signatures; the
/// four-argument form also declares determinism and null behavior.
class AsyncRPCFunctionRegistrar {
 public:
  AsyncRPCFunctionRegistrar(
      const std::string& name,
      AsyncRPCFunctionRegistry::Factory factory,
      AsyncRPCFunctionRegistry::Signatures signatures) {
    AsyncRPCFunctionRegistry::registerFunction(
        name, std::move(factory), std::move(signatures));
  }

  AsyncRPCFunctionRegistrar(
      const std::string& name,
      AsyncRPCFunctionRegistry::Factory factory,
      AsyncRPCFunctionRegistry::Signatures signatures,
      AsyncRPCFunctionRegistry::Metadata metadata) {
    AsyncRPCFunctionRegistry::registerFunction(
        name, std::move(factory), std::move(signatures), std::move(metadata));
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

/// Register an RPC function with a custom factory, explicit signatures, and an
/// explicit stub declaration. Needed when the defaults are wrong for the
/// function -- see AsyncRPCFunctionRegistry::Metadata.
/// Usage: VELOX_REGISTER_RPC_FUNCTION_WITH_METADATA(
///            my_rpc, myFactoryFn, MyClass::signatures(), myMetadata());
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
#define VELOX_REGISTER_RPC_FUNCTION_WITH_METADATA(name, factory, sigs, meta) \
  static ::facebook::velox::exec::rpc::AsyncRPCFunctionRegistrar             \
  _VELOX_RPC_REGISTRAR_VAR2(name, __LINE__)(#name, (factory), (sigs), (meta))

} // namespace facebook::velox::exec::rpc
