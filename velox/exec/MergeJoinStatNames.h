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

#include <string_view>

namespace facebook::velox::exec {

/// Names of the runtime stats reported by MergeJoin.
struct MergeJoinStatNames {
  /// Number of left rows matched in merge join.
  static constexpr std::string_view kMatchedLeftRows = "matchedLeftRows";

  /// Number of right rows matched in merge join.
  static constexpr std::string_view kMatchedRightRows = "matchedRightRows";

  /// Number of left input batches dropped mid-group by
  /// 'merge_join_stream_left_side'. Zero means no group ever streamed, so the
  /// left side was retained exactly as it is without the config.
  static constexpr std::string_view kStreamedLeftBatches =
      "streamedLeftBatches";
};

} // namespace facebook::velox::exec
