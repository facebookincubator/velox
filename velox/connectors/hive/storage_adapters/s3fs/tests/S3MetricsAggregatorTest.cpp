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
#include "velox/connectors/hive/storage_adapters/s3fs/S3Metrics.h"

namespace facebook::velox::filesystems {
namespace {

class S3MetricsAggregatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    aggregator_ = S3MetricsAggregator::getInstance().get();
    ASSERT_NE(aggregator_, nullptr)
        << "Failed to get S3MetricsAggregator instance";
    // Manually reset relevant metrics before each test using the metric names
    // from S3Metrics.h
    aggregator_->resetMetric(kMetricS3ActiveConnections);
    aggregator_->resetMetric(kMetricS3StartedUploads);
    aggregator_->resetMetric(kMetricS3FailedUploads);
  }

  void TearDown() override {
    // Cleanup or reset any global state if necessary
  }

  S3MetricsAggregator* aggregator_;
};

TEST_F(S3MetricsAggregatorTest, IncrementAndRetrieveMetrics) {
  // Ensure the metric starts at 0
  EXPECT_EQ(aggregator_->getMetric(kMetricS3ActiveConnections), 0);

  // Increment the metric and check the value
  aggregator_->incrementMetric(kMetricS3ActiveConnections);
  EXPECT_EQ(aggregator_->getMetric(kMetricS3ActiveConnections), 1);

  // Increment the metric again and check the value
  aggregator_->incrementMetric(kMetricS3ActiveConnections);
  EXPECT_EQ(aggregator_->getMetric(kMetricS3ActiveConnections), 2);
}

TEST_F(S3MetricsAggregatorTest, ResetMetrics) {
  // Increment the metric
  aggregator_->incrementMetric(kMetricS3StartedUploads);
  aggregator_->incrementMetric(kMetricS3StartedUploads);

  // Ensure the metric value is 2
  EXPECT_EQ(aggregator_->getMetric(kMetricS3StartedUploads), 2);

  // Reset the metric and check the value
  aggregator_->resetMetric(kMetricS3StartedUploads);
  EXPECT_EQ(aggregator_->getMetric(kMetricS3StartedUploads), 0);

  // Reset the metric again to verify idempotency
  aggregator_->resetMetric(kMetricS3StartedUploads);
  EXPECT_EQ(aggregator_->getMetric(kMetricS3StartedUploads), 0);
}

TEST_F(S3MetricsAggregatorTest, ThreadSafety) {
  const int numThreads = 10;
  const int incrementsPerThread = 1000;

  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([this]() {
      for (int j = 0; j < incrementsPerThread; ++j) {
        aggregator_->incrementMetric(kMetricS3MetadataCalls);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Check that the final metric value is correct
  ASSERT_EQ(
      aggregator_->getMetric(kMetricS3MetadataCalls),
      numThreads * incrementsPerThread);
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
