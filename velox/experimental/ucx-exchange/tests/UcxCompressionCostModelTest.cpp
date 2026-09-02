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
#include "velox/experimental/ucx-exchange/UcxCompressionCostModel.h"

#include <gtest/gtest.h>

namespace facebook::velox::ucx_exchange::test {

namespace {

void recordSample(
    UcxCompressionCostModel& model,
    std::string_view task,
    std::size_t rawBytes,
    std::size_t candidateBytes,
    double encodeSeconds,
    double transferSeconds,
    double decodeSeconds) {
  model.recordEncode(task, rawBytes, candidateBytes, encodeSeconds);
  model.recordTransfer(task, candidateBytes, transferSeconds);
  model.recordDecode(task, rawBytes, decodeSeconds);
}

} // namespace

TEST(UcxCompressionCostModelTest, ExtractsQueryStageKey) {
  EXPECT_EQ(
      UcxCompressionCostModel::stageKey("20260724_233601_00013_yb67m.8.0.3.0"),
      "20260724_233601_00013_yb67m.8");
  EXPECT_EQ(UcxCompressionCostModel::stageKey("query.4"), "query.4");
  EXPECT_EQ(UcxCompressionCostModel::stageKey("query"), "query");
}

TEST(UcxCompressionCostModelTest, FusesWarmupWithNormalCodecSamples) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.4.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // 60% byte savings at 10 kB/s saves 60 ms; the codec costs 20 ms.
    recordSample(model, task, 1'000, 400, 0.010, 0.040, 0.010);
  }
  const auto decision = model.decide(task, 1'000);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kCompress);
  EXPECT_DOUBLE_EQ(decision.candidateRatio, 0.4);
  EXPECT_DOUBLE_EQ(decision.effectiveTransferBytesPerSecond, 10'000.0);
  EXPECT_NEAR(decision.estimatedTransferSavedSeconds, 0.060, 1e-12);
  EXPECT_NEAR(decision.estimatedCodecSeconds, 0.020, 1e-12);
}

TEST(UcxCompressionCostModelTest, SendsRawAndPeriodicallyReprobes) {
  UcxCompressionCostModel model;
  constexpr std::string_view task{"query.3.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // 20 ms transfer savings cannot repay 50 ms of codec work.
    recordSample(model, task, 1'000, 800, 0.030, 0.080, 0.020);
  }
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kRaw);
  }
  EXPECT_EQ(
      model.decide(task, 1'000).action,
      UcxCompressionCostModel::Action::kProbe);
}

TEST(UcxCompressionCostModelTest, AppliesCodecSafetyMargin) {
  UcxCompressionCostModel model(4, 8, 1.5);
  constexpr std::string_view task{"query.5.0.0.0"};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        model.decide(task, 1'000).action,
        UcxCompressionCostModel::Action::kProbe);
    // Saving 60 ms is profitable against 50 ms of codec work without a
    // margin, but not after reserving 50% for opportunity cost.
    recordSample(model, task, 1'000, 400, 0.030, 0.040, 0.020);
  }
  const auto decision = model.decide(task, 1'000);
  EXPECT_NEAR(decision.estimatedTransferSavedSeconds, 0.060, 1e-12);
  EXPECT_NEAR(decision.estimatedCodecSeconds, 0.050, 1e-12);
  EXPECT_EQ(decision.action, UcxCompressionCostModel::Action::kRaw);
}

TEST(UcxCompressionCostModelTest, SharesDecodeRateWithinAProcess) {
  UcxCompressionCostModel model;
  constexpr std::string_view incoming{"query.8.0.3.0"};
  constexpr std::string_view outgoing{"query.8.0.0.0"};
  model.recordDecode(incoming, 1'000, 0.010);
  model.recordDecode(incoming, 1'000, 0.010);
  for (int i = 0; i < 4; ++i) {
    model.recordEncode(outgoing, 1'000, 400, 0.010);
    model.recordTransfer(outgoing, 400, 0.040);
  }
  EXPECT_EQ(
      model.decide(outgoing, 1'000).action,
      UcxCompressionCostModel::Action::kCompress);
}

} // namespace facebook::velox::ucx_exchange::test
