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
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace facebook::velox::ucx_exchange {

/// Online cost model for the opt-in "column-adaptive" exchange codec.
///
/// A probe is an ordinary column-codec invocation. It simultaneously chooses
/// the per-region codec, measures encode time, and produces the candidate that
/// is sent. Transfer completion and the receiver's real decode complete the
/// same sample; this model never asks for a duplicate encode or local decode.
class UcxCompressionCostModel {
 public:
  enum class Action {
    kProbe,
    kCompress,
    kRaw,
  };

  struct Decision {
    Action action{Action::kProbe};
    std::size_t encodeSamples{0};
    std::size_t transferSamples{0};
    std::size_t decodeSamples{0};
    double candidateRatio{0.0};
    double effectiveTransferBytesPerSecond{0.0};
    double estimatedTransferSavedSeconds{0.0};
    double estimatedCodecSeconds{0.0};
  };

  explicit UcxCompressionCostModel(
      std::size_t warmupSamples = 4,
      std::size_t reprobeInterval = 8,
      double codecSafetyMargin = 1.10,
      std::size_t windowSamples = 8);

  /// codecSafetyMargin is captured on the first call. Worker configuration is
  /// immutable before exchange processing begins, so subsequent calls must use
  /// the same value.
  static UcxCompressionCostModel& instance(double codecSafetyMargin = 1.10);

  /// Chooses whether the next eligible chunk should run the normal column
  /// codec. kProbe and kCompress both run exactly the same codec path.
  Decision decide(std::string_view taskId, std::size_t rawBytes);

  /// Records the result of the normal column codec invocation, including
  /// candidates that the codec rejects on its intrinsic byte-savings check.
  void recordEncode(
      std::string_view taskId,
      std::size_t rawBytes,
      std::size_t candidateBytes,
      double seconds);

  /// Records the real UCX send-completion interval. Raw sends are useful
  /// observations too and keep the transport estimate current.
  void recordTransfer(
      std::string_view taskId,
      std::size_t wireBytes,
      double seconds);

  /// Records a real receiver-side column decode. Each worker both sends and
  /// receives exchange data, so receiver observations seed the matching
  /// query-stage model in the same process without protocol feedback.
  void recordDecode(
      std::string_view remoteTaskId,
      std::size_t rawBytes,
      double seconds);

  static std::string stageKey(std::string_view taskId);
  static std::string_view actionName(Action action);

 private:
  struct ByteTimeSample {
    std::size_t bytes{0};
    double seconds{0.0};
  };

  struct EncodeSample {
    std::size_t rawBytes{0};
    std::size_t candidateBytes{0};
    double seconds{0.0};
  };

  struct StageSamples {
    std::deque<EncodeSample> encodes;
    std::deque<ByteTimeSample> transfers;
    std::deque<ByteTimeSample> decodes;
    std::size_t rawDecisionsSinceProbe{0};
  };

  template <typename T>
  void append(std::deque<T>& samples, T sample) {
    samples.push_back(std::move(sample));
    while (samples.size() > windowSamples_) {
      samples.pop_front();
    }
  }

  const std::size_t warmupSamples_;
  const std::size_t reprobeInterval_;
  const double codecSafetyMargin_;
  const std::size_t windowSamples_;

  std::mutex mutex_;
  std::unordered_map<std::string, StageSamples> stages_;
  std::deque<ByteTimeSample> globalDecodes_;
};

} // namespace facebook::velox::ucx_exchange
