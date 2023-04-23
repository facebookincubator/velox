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
#include "velox/type/Uuid.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;

class UuidTest : public functions::test::FunctionBaseTest {};

TEST_F(UuidTest, testUuid) {
  auto uuid1 = evaluateOnce<Uuid, bool>("uuid()", true);
  auto uuid2 = evaluateOnce<Uuid, bool>("uuid()", true);
  ASSERT_NE(uuid1.value().id(), uuid2.value().id());
}
