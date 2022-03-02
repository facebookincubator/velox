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
#include <optional>
#include <vector>

#include "velox/experimental/udf_adapters/EagerContainiersWrappers.h"
#include "velox/type/Type.h"

// Resolver, exec environment and adapter to run Velox UDFs on eager containers.
// more specifically environment Array is represented as std::vector and maps
// are represented using F14FastMap inputs.

namespace facebook::velox::utils {

template <typename T>
struct BackwardResolver {
  using in_type = T;
  using out_type = T;
};

template <typename T>
struct BackwardResolver<velox::Array<T>> {
  using in_type = std::vector<T>;
  using out_type = VectorWrapper<std::vector<T>>;
};

struct BackwardExec {
  template <typename T>
  using resolver = BackwardResolver<T>;
};

// This functions run a velox UDF `Func` on the system described above.
// The way to use it is simple:
//    utils::UDFWrapper<Func, std::vector<int64_t>, const int64_t&>(input[0]);
// We can potential pass Array<int64_t> and int64_t instead (velox types).
// The function above will run Velox UDF and returns std::vector<int64_t> as an
// output. complex inputs are represented using  std::vector and F14FastMap.
template <template <class> typename Func, typename Return, typename... Args>
Return UDFWrapper(Args&&... args) {
  using UDF = Func<BackwardExec>;
  UDF instance;

  Return result;
  auto resultWrapped = typename WrapperTypeResolver<Return>::type(result);

  // TODO: support callNullable
  bool hasValue = instance.call(resultWrapped, std::forward<Args>(args)...);
  VELOX_CHECK(hasValue);
  return result;
}

}; // namespace facebook::velox::utils