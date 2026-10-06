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

#include <gtest/gtest.h>

#include "velox/common/text/TextFieldParser.h"

namespace facebook::velox::text {
namespace {

TEST(TextFieldParserTest, booleanOneZeroOption) {
  EXPECT_EQ(TextFieldParser::parseBoolean("1", true), true);
  EXPECT_EQ(TextFieldParser::parseBoolean("0", true), false);
  EXPECT_FALSE(TextFieldParser::parseBoolean("1", false).has_value());
  EXPECT_FALSE(TextFieldParser::parseBoolean("0", false).has_value());
}

} // namespace
} // namespace facebook::velox::text
