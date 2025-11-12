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

#include "velox/exec/tests/SpillerBenchmarkBase.h"

namespace facebook::velox::exec::test {
/// This test measures the spill input overhead in spill join & probe.
class JoinSpillInputBenchmarkBase : public SpillerBenchmarkBase {
 public:
  JoinSpillInputBenchmarkBase() = default;

  /// Sets up the test.
  void setUp(
      RowTypePtr rowType =
          ROW({"c0", "c1", "c2", "c3", "c4"},
              {INTEGER(), BIGINT(), VARCHAR(), VARBINARY(), DOUBLE()}),
      int32_t stringMaxLength = 10) override;

  /// Runs the test.
  void run() override;
};
} // namespace facebook::velox::exec::test
