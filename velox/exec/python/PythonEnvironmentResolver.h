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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/ScopedRegistry.h"

namespace facebook::velox::exec::python {

/// How to start a Python interpreter for an environment. A resolver promises
/// that running
///
///   <path> <arguments...> <script> <script arguments...>
///
/// runs <script> as __main__ with the script arguments in sys.argv and the
/// environment variables set, and that pyarrow imports. A plain interpreter
/// does this natively; a program that bundles packages can act as one.
struct PythonInterpreter {
  /// Executable to run: an interpreter, or a program that acts as one.
  std::string path;

  /// Arguments before the script to run.
  std::vector<std::string> arguments;

  /// Environment variables to add to the process's own, as NAME=VALUE.
  std::vector<std::string> environmentVariables;
};

/// Resolves the environment URIs of one scheme into interpreters.
class PythonEnvironmentResolver {
 public:
  virtual ~PythonEnvironmentResolver() = default;

  /// Returns how to start the interpreter for 'uri', whose scheme is the one
  /// this resolver is registered for. May block to fetch the environment.
  /// Thread-safe.
  virtual PythonInterpreter resolveUri(std::string_view uri) const = 0;
};

/// Resolves file://<interpreter path> to the interpreter at that path.
class FileEnvironmentResolver : public PythonEnvironmentResolver {
 public:
  PythonInterpreter resolveUri(std::string_view uri) const override;
};

/// Python environment resolvers by URI scheme. A query that runs Python calls
/// sets its resolvers on its QueryCtx under kRegistryKey:
///
///   auto resolvers = std::make_shared<PythonEnvironmentResolvers>();
///   resolvers->insert("file", std::make_shared<FileEnvironmentResolver>());
///   queryCtx->setRegistry(PythonEnvironmentResolvers::kRegistryKey,
///   resolvers);
class PythonEnvironmentResolvers
    : public ScopedRegistry<std::string, const PythonEnvironmentResolver> {
 public:
  using ScopedRegistry::ScopedRegistry;

  /// The key of a query's registry on its QueryCtx.
  static constexpr std::string_view kRegistryKey{"pythonEnvironmentResolvers"};

  /// Resolves 'uri' with the resolver registered for its scheme. Fails with a
  /// user error for a URI with no scheme or a scheme with no resolver.
  PythonInterpreter resolve(std::string_view uri) const;
};

} // namespace facebook::velox::exec::python
