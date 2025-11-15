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

#include <string>
#include "velox/functions/prestosql/S2Functions.h"

namespace facebook::velox::functions {

void registerS2Functions(const std::string& prefix) {
  registerS2CellIdParent(prefix);
  registerS2CellAreaSqKm(prefix);
  registerS2CellTokenParent(prefix);
}

} // namespace facebook::velox::functions
