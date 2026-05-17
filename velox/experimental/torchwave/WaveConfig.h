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

#include <cstdint>
#include <optional>

namespace torch::wave {

struct WaveConfig {
  static constexpr int32_t kNodes = 1;
  static constexpr int32_t kLaunches = 2;
  static constexpr int32_t kTensors = 4;
  static constexpr int32_t kFrame = 8;

  int32_t blockSize{256};
  int32_t singleBlockPathBlockSize{1024};
  bool allStandalone{false};
  int32_t numStandaloneThreads{0};

  /// If non-zero, use this as the number of SMs instead of reading from the
  /// device.
  int32_t numSms{0};

  /// Trace bit mask. kNodes prints node headers, kLaunches prints per-launch
  /// details.
  int32_t trace{0};

  /// If set, forces the grid choice between single-block and multi-block
  /// variants. If nullopt, the choice is made based on input size.
  std::optional<bool> useSingleBlock;

  /// If set and true, use the cooperative grid variant when available.
  std::optional<bool> isCg;

  static WaveConfig& get() {
    static WaveConfig instance;
    return instance;
  }
};

} // namespace torch::wave
