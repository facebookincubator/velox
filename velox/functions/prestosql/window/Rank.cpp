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
#include "velox/common/base/Exceptions.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/window/WindowFunctionNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window {

namespace {

class RankFunction : public exec::WindowFunction {
 public:
  explicit RankFunction() : WindowFunction(BIGINT()) {}

  void resetPartition(const std::vector<char*>& /*rows*/) {
    rank_ = 1;
    currentPeerGroupStart_ = 0;
    previousPeerCount_ = 0;
  }

  void apply(
      int32_t peerGroupStarts,
      int32_t peerGroupEnds,
      int32_t /* frameStarts */,
      int32_t /* frameEnds */,
      int32_t currentOutputRow,
      const std::vector<VectorPtr>& /* argVectors */,
      const VectorPtr& result) {
    if (peerGroupStarts != currentPeerGroupStart_) {
      currentPeerGroupStart_ = peerGroupStarts;
      rank_ += previousPeerCount_;
    }
    previousPeerCount_ += 1;
    result->asFlatVector<int64_t>()->mutableRawValues()[currentOutputRow] =
        rank_;
  }

 private:
  int32_t currentPeerGroupStart_ = 0;
  int32_t previousPeerCount_ = 0;
  int64_t rank_ = 1;
};

bool registerRank(const std::string& name) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures{
      exec::FunctionSignatureBuilder().returnType("bigint").build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<TypePtr>& argTypes, const TypePtr&
          /*resultType*/) -> std::unique_ptr<exec::WindowFunction> {
        VELOX_CHECK_LE(argTypes.size(), 0, "{} takes no arguments", name);
        return std::make_unique<RankFunction>();
      });
  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_WindowFunction) = registerRank(kRank);

} // namespace
} // namespace facebook::velox::window
