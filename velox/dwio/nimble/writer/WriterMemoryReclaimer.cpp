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
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::nimble {

// Drives the writer's reclaim path during memory arbitration. Derives from
// velox::exec::MemoryReclaimer rather than the velox::memory base so that
// enterArbitration()/leaveArbitration() suspend the Velox Driver when the
// request originates on a driver thread, which is the case for the connector
// write path.
//
// Kept at namespace scope (not an anonymous namespace) so Writer can befriend
// it by name to reach its private reclaimableBytes()/reclaimBytes().
// @lint-ignore CLANGTIDY facebook-hte-ShadowingClass
class WriterMemoryReclaimer : public velox::exec::MemoryReclaimer {
 public:
  static std::unique_ptr<velox::memory::MemoryReclaimer> create(
      Writer* writer) {
    return std::unique_ptr<velox::memory::MemoryReclaimer>(
        new WriterMemoryReclaimer(writer));
  }

  bool reclaimableBytes(
      const velox::memory::MemoryPool& pool,
      uint64_t& reclaimableBytes) const override {
    return writer_->reclaimableBytes(pool, reclaimableBytes);
  }

  uint64_t reclaim(
      velox::memory::MemoryPool* pool,
      uint64_t /*targetBytes*/,
      uint64_t /*maxWaitMs*/,
      velox::memory::MemoryReclaimer::Stats& stats) override {
    return writer_->reclaimBytes(pool, stats);
  }

 private:
  explicit WriterMemoryReclaimer(Writer* writer)
      : velox::exec::MemoryReclaimer(0), writer_{writer} {
    NIMBLE_CHECK_NOT_NULL(writer_);
  }

  Writer* const writer_;
};

std::unique_ptr<velox::memory::MemoryReclaimer> Writer::makeMemoryReclaimer() {
  return WriterMemoryReclaimer::create(this);
}

} // namespace facebook::nimble
