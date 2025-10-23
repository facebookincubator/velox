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

#include "SpillerBenchmarkBase.h"

namespace facebook::velox::exec::test {
/// This test measures the spill input overhead in spill join & probe.
class SortSpillInputBenchmarkBase : public SpillerBenchmarkBase {
public:
 SortSpillInputBenchmarkBase() = default;

 void setUp(RowTypePtr rowType, int32_t stringMaxLength) override {
   setUp(rowType, false, stringMaxLength);
 };
 /// Sets up the test.
 void setUp(
     RowTypePtr rowType,
     bool serializeRowContainer,
     int32_t stringMaxLength);

 /// Runs the test.
 void run() override;

 void printStats() const override;

private:
 void writeSpillData();
 std::unique_ptr<SpillerBase> makeSpiller();

 const std::string spillerType_;
 std::unique_ptr<RowContainer> rowContainer_;
 std::shared_ptr<velox::memory::MemoryPool> spillerPool_;
 bool serializeRowContainer_;
};
} // namespace facebook::velox::exec::test