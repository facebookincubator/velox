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

#include "velox/experimental/udf_adapters/BackwardAdapter.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace facebook::velox {
class BackwardAdapterTest : public functions::test::FunctionBaseTest {};

// Function that creates array with values 0...n-1.
// Uses all possible functions in the array proxy interface.
template <typename T>
struct Func {
  template <typename TOut>
  bool call(TOut& out, const int64_t& n) {
    for (int i = 0; i < n; i++) {
      out.add_item() = i;
    }
    return true;
  }
};

TEST_F(BackwardAdapterTest, testSimpleArray) {
  // Run the function in velox.
  registerFunction<Func, ArrayProxyT<int64_t>, int64_t>({"func"});
  std::vector<int64_t> input = {10};

  auto flatVector = makeFlatVector<int64_t>(input);
  auto resultVelox =
      evaluate<ArrayVector>("func(c0)", makeRowVector({flatVector}));
  auto resultStd = utils::UDFWrapper<Func, std::vector<int64_t>, int64_t>(10);

  // Reader for velox results.
  DecodedVector decoded;
  SelectivityVector rows(resultVelox->size());
  decoded.decode(*resultVelox, rows);
  exec::VectorReader<Array<int64_t>> reader(&decoded);
  auto arrayView = reader[0];

  // Assert same results.
  for (auto i = 0; i < 10; i++) {
    ASSERT_EQ(arrayView[i].value(), resultStd[i]);
  }
}
} // namespace facebook::velox
