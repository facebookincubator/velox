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

#include <algorithm>
#include <numeric>

namespace facebook::velox::ucx_exchange {

namespace {

constexpr std::size_t kMinimumDecodeSamples = 2;

template <typename Samples, typename Value>
Value sumBytes(const Samples& samples, Value value) {
  for (const auto& sample : samples) {
    value += sample.bytes;
  }
  return value;
}

template <typename Samples>
double sumSeconds(const Samples& samples) {
  return std::accumulate(
      samples.begin(),
      samples.end(),
      0.0,
      [](double total, const auto& sample) { return total + sample.seconds; });
}

} // namespace

UcxCompressionCostModel::UcxCompressionCostModel(
    std::size_t warmupSamples,
    std::size_t reprobeInterval,
    double codecSafetyMargin,
    std::size_t windowSamples)
    : warmupSamples_(std::max<std::size_t>(1, warmupSamples)),
      reprobeInterval_(std::max<std::size_t>(1, reprobeInterval)),
      codecSafetyMargin_(std::max(1.0, codecSafetyMargin)),
      windowSamples_(
          std::max({std::size_t{1}, windowSamples, warmupSamples_})) {}

UcxCompressionCostModel& UcxCompressionCostModel::instance(
    double codecSafetyMargin) {
  static UcxCompressionCostModel model(4, 8, codecSafetyMargin);
  return model;
}

UcxCompressionCostModel::Decision UcxCompressionCostModel::decide(
    std::string_view taskId,
    std::size_t rawBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& stage = stages_[stageKey(taskId)];

  Decision decision;
  decision.encodeSamples = stage.encodes.size();
  decision.transferSamples = stage.transfers.size();

  const auto& decodes = stage.decodes.size() >= kMinimumDecodeSamples
      ? stage.decodes
      : globalDecodes_;
  decision.decodeSamples = decodes.size();

  if (stage.encodes.size() < warmupSamples_ ||
      stage.transfers.size() < warmupSamples_ ||
      decodes.size() < kMinimumDecodeSamples || rawBytes == 0) {
    stage.rawDecisionsSinceProbe = 0;
    return decision;
  }

  std::size_t encodedRawBytes = 0;
  std::size_t candidateBytes = 0;
  double encodeSeconds = 0.0;
  for (const auto& sample : stage.encodes) {
    encodedRawBytes += sample.rawBytes;
    candidateBytes += sample.candidateBytes;
    encodeSeconds += sample.seconds;
  }
  const auto transferredBytes = sumBytes(stage.transfers, std::size_t{0});
  const auto transferSeconds = sumSeconds(stage.transfers);
  const auto decodedBytes = sumBytes(decodes, std::size_t{0});
  const auto decodeSeconds = sumSeconds(decodes);

  if (encodedRawBytes == 0 || transferredBytes == 0 || transferSeconds <= 0.0 ||
      decodedBytes == 0 || decodeSeconds <= 0.0) {
    stage.rawDecisionsSinceProbe = 0;
    return decision;
  }

  decision.candidateRatio =
      static_cast<double>(candidateBytes) / encodedRawBytes;
  decision.effectiveTransferBytesPerSecond =
      static_cast<double>(transferredBytes) / transferSeconds;
  const auto savedBytes =
      std::max(0.0, rawBytes * (1.0 - decision.candidateRatio));
  decision.estimatedTransferSavedSeconds =
      savedBytes / decision.effectiveTransferBytesPerSecond;
  decision.estimatedCodecSeconds = rawBytes *
      (encodeSeconds / encodedRawBytes + decodeSeconds / decodedBytes);

  if (decision.estimatedTransferSavedSeconds >
      codecSafetyMargin_ * decision.estimatedCodecSeconds) {
    stage.rawDecisionsSinceProbe = 0;
    decision.action = Action::kCompress;
    return decision;
  }

  ++stage.rawDecisionsSinceProbe;
  if (stage.rawDecisionsSinceProbe >= reprobeInterval_) {
    stage.rawDecisionsSinceProbe = 0;
    decision.action = Action::kProbe;
    return decision;
  }
  decision.action = Action::kRaw;
  return decision;
}

void UcxCompressionCostModel::recordEncode(
    std::string_view taskId,
    std::size_t rawBytes,
    std::size_t candidateBytes,
    double seconds) {
  if (rawBytes == 0 || candidateBytes == 0 || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  append(
      stages_[stageKey(taskId)].encodes,
      EncodeSample{rawBytes, candidateBytes, seconds});
}

void UcxCompressionCostModel::recordTransfer(
    std::string_view taskId,
    std::size_t wireBytes,
    double seconds) {
  if (wireBytes == 0 || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  append(
      stages_[stageKey(taskId)].transfers, ByteTimeSample{wireBytes, seconds});
}

void UcxCompressionCostModel::recordDecode(
    std::string_view remoteTaskId,
    std::size_t rawBytes,
    double seconds) {
  if (rawBytes == 0 || seconds <= 0.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const ByteTimeSample sample{rawBytes, seconds};
  append(stages_[stageKey(remoteTaskId)].decodes, sample);
  append(globalDecodes_, sample);
}

std::string UcxCompressionCostModel::stageKey(std::string_view taskId) {
  const auto first = taskId.find('.');
  if (first == std::string_view::npos) {
    return std::string(taskId);
  }
  const auto second = taskId.find('.', first + 1);
  if (second == std::string_view::npos) {
    return std::string(taskId);
  }
  return std::string(taskId.substr(0, second));
}

std::string_view UcxCompressionCostModel::actionName(Action action) {
  switch (action) {
    case Action::kProbe:
      return "probe";
    case Action::kCompress:
      return "compress";
    case Action::kRaw:
      return "raw";
  }
  return "unknown";
}

} // namespace facebook::velox::ucx_exchange
