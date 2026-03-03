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
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/StringFunctions.h"
#include "velox/functions/prestosql/VarbinaryFunctions.h"

namespace facebook::velox::functions {

void registerVarbinaryFunctions(const std::string& prefix) {
  // Length for varbinary
  registerFunction<LengthVarbinaryFunction, int64_t, Varbinary>(
      {prefix + "length"});

  // substr for varbinary
  registerFunction<SubstrVarbinaryFunction, Varbinary, Varbinary, int64_t>(
      {prefix + "substr"});
  registerFunction<
      SubstrVarbinaryFunction,
      Varbinary,
      Varbinary,
      int64_t,
      int64_t>({prefix + "substr"});

  // strpos/strrpos for varbinary
  registerFunction<StrLPosVarbinaryFunction, int64_t, Varbinary, Varbinary>(
      {prefix + "strpos"});
  registerFunction<
      StrLPosVarbinaryFunction,
      int64_t,
      Varbinary,
      Varbinary,
      int64_t>({prefix + "strpos"});
  registerFunction<StrRPosVarbinaryFunction, int64_t, Varbinary, Varbinary>(
      {prefix + "strrpos"});
  registerFunction<
      StrRPosVarbinaryFunction,
      int64_t,
      Varbinary,
      Varbinary,
      int64_t>({prefix + "strrpos"});

  // contains for varbinary
  registerFunction<ContainsVarbinaryFunction, bool, Varbinary, Varbinary>(
      {prefix + "contains"});
}
} // namespace facebook::velox::functions
