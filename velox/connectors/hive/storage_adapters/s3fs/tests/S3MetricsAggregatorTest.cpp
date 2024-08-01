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

#include "velox/connectors/hive/storage_adapters/s3fs/S3MetricsAggregator.h"
#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace facebook::velox::filesystems {
namespace {

class S3MetricsAggregatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    aggregator_ = S3MetricsAggregator::getInstance().get();
    aggregator_->resetMetric("test_metric"); // Reset metric before each test
  }

  S3MetricsAggregator* aggregator_;
};

TEST_F(S3MetricsAggregatorTest, IncrementAndRetrieveMetrics) {
  const std::string metricName = "test_metric";

  // Ensure the metric starts at 0
  EXPECT_EQ(aggregator_->getMetric(metricName), 0);

  // Increment the metric and check the value
  aggregator_->incrementMetric(metricName);
  EXPECT_EQ(aggregator_->getMetric(metricName), 1);

  // Increment the metric again and check the value
  aggregator_->incrementMetric(metricName);
  EXPECT_EQ(aggregator_->getMetric(metricName), 2);
}

TEST_F(S3MetricsAggregatorTest, ResetMetrics) {
  const std::string metricName = "test_metric";

  // Increment the metric
  aggregator_->incrementMetric(metricName);
  aggregator_->incrementMetric(metricName);

  // Ensure the metric value is 2
  EXPECT_EQ(aggregator_->getMetric(metricName), 2);

  // Reset the metric and check the value
  aggregator_->resetMetric(metricName);
  EXPECT_EQ(aggregator_->getMetric(metricName), 0);
}

} // namespace
} // namespace facebook::velox::filesystems

int main(int argc, char** argv) {
  // Initialize folly and gtest
  folly::Init init{&argc, &argv, false};
  testing::InitGoogleTest(&argc, argv);

  // Ensure google logging is initialized only once
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(argv[0]);
  }

  return RUN_ALL_TESTS();
}
