/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <folly/executors/InlineExecutor.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/writer/WriterFactory.h"

namespace facebook::velox::nimble {
namespace {

class CloneableFormatOptions : public dwio::common::FormatSpecificOptions {
 public:
  explicit CloneableFormatOptions(int32_t value) : value{value} {}

  std::shared_ptr<dwio::common::FormatSpecificOptions> clone() const override {
    return std::make_shared<CloneableFormatOptions>(*this);
  }

  int32_t value;
};

TEST(NimbleWriterOptionsTest, clonePreservesTypeAndIsolatesMutableState) {
  NimbleWriterOptions options;
  options.encodingExecutor =
      folly::getKeepAliveToken(folly::InlineExecutor::instance());
  options.schemaAttributes[1] = {{"iceberg.id", "7"}};
  options.serdeParameters = {{"key", "value"}};
  options.formatSpecificOptions = std::make_shared<CloneableFormatOptions>(11);

  const auto cloned =
      std::dynamic_pointer_cast<NimbleWriterOptions>(options.clone());
  ASSERT_NE(cloned, nullptr);
  EXPECT_NE(cloned.get(), &options);
  EXPECT_EQ(cloned->encodingExecutor.get(), options.encodingExecutor.get());
  EXPECT_EQ(cloned->schemaAttributes, options.schemaAttributes);
  EXPECT_EQ(cloned->serdeParameters, options.serdeParameters);

  ASSERT_NE(cloned->formatSpecificOptions, nullptr);
  EXPECT_NE(cloned->formatSpecificOptions, options.formatSpecificOptions);
  const auto clonedFormatOptions =
      std::dynamic_pointer_cast<CloneableFormatOptions>(
          cloned->formatSpecificOptions);
  ASSERT_NE(clonedFormatOptions, nullptr);
  EXPECT_EQ(clonedFormatOptions->value, 11);

  cloned->schemaAttributes[1][0].second = "9";
  cloned->serdeParameters["key"] = "changed";
  clonedFormatOptions->value = 13;

  EXPECT_EQ(options.schemaAttributes.at(1)[0].second, "7");
  EXPECT_EQ(options.serdeParameters.at("key"), "value");
  EXPECT_EQ(
      std::dynamic_pointer_cast<CloneableFormatOptions>(
          options.formatSpecificOptions)
          ->value,
      11);
}

} // namespace
} // namespace facebook::velox::nimble
