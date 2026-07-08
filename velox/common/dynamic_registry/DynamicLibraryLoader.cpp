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

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

void loadDynamicLibrary(
    const std::string& fileName,
    const std::string& registrationFunctionName) {
#ifdef _WIN32
  // Windows: Use LoadLibrary/GetProcAddress
  HMODULE handler = LoadLibraryA(fileName.c_str());

  if (handler == nullptr) {
    VELOX_USER_FAIL(
        "Error while loading shared library: {} (error code {})",
        fileName,
        GetLastError());
  }

  LOG(INFO) << "Loaded library " << fileName << ". Searching registry symbol "
            << registrationFunctionName;

  auto loadUserLibrary = reinterpret_cast<void (*)()>(
      GetProcAddress(handler, registrationFunctionName.c_str()));

  if (loadUserLibrary == nullptr) {
    VELOX_USER_FAIL(
        "Couldn't find Velox registry symbol '{}' (error code {})",
        registrationFunctionName,
        GetLastError());
  }

  // Invoke the registry function.
  loadUserLibrary();
  LOG(INFO) << "Registered functions by " << registrationFunctionName;
#else
  // Try to dynamically load the shared library.
  void* handler = dlopen(fileName.c_str(), RTLD_NOW);

  if (handler == nullptr) {
    VELOX_USER_FAIL("Error while loading shared library: {}", dlerror());
  }

  LOG(INFO) << "Loaded library " << fileName << ". Searching registry symbol "
            << registrationFunctionName;

  // Lookup the symbol.
  void* registrySymbol = dlsym(handler, registrationFunctionName.c_str());
  auto loadUserLibrary = reinterpret_cast<void (*)()>(registrySymbol);
  const char* error = dlerror();

  // Check for an error first as a null symbol pointer is not necessarily an
  // error.
  if (error != nullptr) {
    VELOX_USER_FAIL("Couldn't find Velox registry symbol: {}", error);
  }

  if (loadUserLibrary == nullptr) {
    VELOX_USER_FAIL(
        "Symbol '{}' resolved to a nullptr, unable to invoke it.",
        registrationFunctionName);
  }

  // Invoke the registry function.
  loadUserLibrary();
  LOG(INFO) << "Registered functions by " << registrationFunctionName;
#endif
}

} // namespace facebook::velox
