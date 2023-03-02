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

#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::memory {

class MemoryPool;

class MemoryArbitrator {
 public:
  enum class Kind {
    kFixed,
    kShared,
  };

  struct Config {
    Kind kind;
    int64_t capacity;
  };
  static std::unique_ptr<MemoryArbitrator> create(const Config& config);

  Kind kind() const {
    return kind_;
  }
  static std::string kindString(Kind kind);

  virtual ~MemoryArbitrator() = default;

  virtual int64_t reserveMemory(int64_t targetBytes) = 0;
  virtual void releaseMemory(MemoryPool* requestor) = 0;

  virtual bool growMemory(
      MemoryPool* requestor,
      const std::vector<MemoryPool*>& candidates,
      uint64_t targetBytes) = 0;

  struct Stats {
    uint64_t numRequests{0};
    uint64_t numArbitrations{0};
    uint64_t numFailures{0};
    uint64_t numRetries{0};
    uint64_t numShrinkedBytes{0};
    uint64_t numReclaimedBytes{0};
    uint64_t numWaits{0};
    uint64_t waitTimeUs{0};
    uint64_t arbitrationTimeUs{0};

    std::string toString() const;
  };
  virtual Stats stats() const = 0;

  virtual std::string toString() const = 0;

 protected:
  explicit MemoryArbitrator(const Config& config)
      : kind_(config.kind), capacity_(config.capacity) {}

  const Kind kind_;
  const uint64_t capacity_;

  Stats stats_;
};

std::ostream& operator<<(std::ostream& out, const MemoryArbitrator::Kind& kind);

class FixedMemoryArbitrator : public MemoryArbitrator {
 public:
  FixedMemoryArbitrator(const Config& config) : MemoryArbitrator(config) {
    VELOX_CHECK_EQ(kind_, Kind::kFixed);
  }

  virtual int64_t reserveMemory(int64_t targetBytes) final;

  void releaseMemory(MemoryPool* /*unused*/) final {}

  bool growMemory(
      MemoryPool* requestor,
      const std::vector<MemoryPool*>& candidates,
      uint64_t targetBytes) final {
    return false;
  }

  Stats stats() const final {
    // TODO: add implementation later.
    return {};
  }

  std::string toString() const final;
};

class MemoryReclaimer {
 public:
  virtual ~MemoryReclaimer() = default;

  static std::shared_ptr<MemoryReclaimer> create();

  virtual void enterArbitration(){};
  virtual void leaveArbitration() noexcept {};

  virtual bool canReclaim(const MemoryPool& pool) const;
  virtual uint64_t reclaimableBytes(const MemoryPool& pool, bool& reclaimable)
      const;
  virtual uint64_t reclaim(MemoryPool* pool, uint64_t targetBytes);

 protected:
  MemoryReclaimer() = default;
};
} // namespace facebook::velox::memory
