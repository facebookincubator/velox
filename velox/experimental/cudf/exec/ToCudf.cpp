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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/exec/Operator.h" // Compilation fails in Driver.h if Operator.h isn't included first!
#include "velox/exec/Driver.h"

#include <iostream>

namespace facebook::velox::cudf_velox {

bool cudfDriverAdapter(
    const exec::DriverFactory& factory,
    exec::Driver& driver) {
  std::cout << "Calling cudfDriverAdapter" << std::endl;
  return false;
}

void registerCudf() {
  std::cout << "Registering cudfDriverAdapter" << std::endl;
  exec::DriverAdapter cudfAdapter{"cuDF", {}, cudfDriverAdapter};
  exec::DriverFactory::registerAdapter(cudfAdapter);
}
} // namespace facebook::velox::cudf_velox
