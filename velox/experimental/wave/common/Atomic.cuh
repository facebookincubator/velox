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

#include <breeze/platforms/platform.h>
#include <breeze/utils/types.h>
#include <breeze/platforms/cuda.cuh>
#include "velox/experimental/wave/common/BitUtil.cuh"
#include "velox/experimental/wave/common/CompilerDefines.h"

namespace facebook::velox::wave {

enum class MemoryScope { kDevice };

enum class MemoryOrder {
  kRelaxed,
  kAcquire,
  kRelease,
};

template <MemoryScope>
struct MemoryScopeTraits;

template <>
struct MemoryScopeTraits<MemoryScope::kDevice> {
  WAVE_DEVICE_HOST static constexpr breeze::utils::AddressSpace
  breeze_address_space() {
    return breeze::utils::GLOBAL;
  }
};

template <MemoryOrder>
struct MemoryOrderTraits;

template <>
struct MemoryOrderTraits<MemoryOrder::kRelaxed> {
  WAVE_DEVICE_HOST static constexpr breeze::utils::MemoryOrder breeze_type() {
    return breeze::utils::RELAXED;
  }
};

template <>
struct MemoryOrderTraits<MemoryOrder::kAcquire> {
  WAVE_DEVICE_HOST static constexpr breeze::utils::MemoryOrder breeze_type() {
    return breeze::utils::ACQUIRE;
  }
};

template <>
struct MemoryOrderTraits<MemoryOrder::kRelease> {
  WAVE_DEVICE_HOST static constexpr breeze::utils::MemoryOrder breeze_type() {
    return breeze::utils::RELEASE;
  }
};

template <typename T, MemoryScope Scope = MemoryScope::kDevice>
struct Atomic {
  // Note: We currently do not require usage of this class to specify
  // kBlockThreads as that is not technically used by the CUDA platform
  // implementation of the atomics in breeze and is non-trivial for
  // existing usage of atomics to provide. We might want to change that
  // in the future to remove that assumption.
  using PlatformT = CudaPlatform</*kBlockThreads=*/kWarpThreads, kWarpThreads>;

  alignas(sizeof(T)) T __value;

  Atomic(const Atomic&) = delete;
  Atomic& operator=(const Atomic&) = delete;
  Atomic& operator=(const Atomic&) volatile = delete;

  WAVE_DEVICE_HOST explicit constexpr Atomic() noexcept = default;
  WAVE_DEVICE_HOST constexpr explicit inline Atomic(T value) noexcept
      : __value(value) {}

  template <MemoryOrder Order = MemoryOrder::kRelaxed>
  WAVE_DEVICE_HOST T load() noexcept {
    using namespace breeze::utils;
    return PlatformT().atomic_load<MemoryOrderTraits<Order>::breeze_type()>(
        make_slice<MemoryScopeTraits<Scope>::breeze_address_space()>(&__value));
  }

  template <MemoryOrder Order = MemoryOrder::kRelaxed>
  WAVE_DEVICE_HOST void store(T value) noexcept {
    using namespace breeze::utils;
    PlatformT().atomic_store<MemoryOrderTraits<Order>::breeze_type()>(
        make_slice<MemoryScopeTraits<Scope>::breeze_address_space()>(&__value),
        value);
  }

  template <MemoryOrder Order = MemoryOrder::kRelaxed>
  WAVE_DEVICE_HOST bool compare_exchange(T& expected, T desired) noexcept {
    using namespace breeze::utils;
    T actual = PlatformT().atomic_cas<MemoryOrderTraits<Order>::breeze_type()>(
        make_slice<MemoryScopeTraits<Scope>::breeze_address_space()>(&__value),
        expected,
        desired);
    bool was_changed = actual == expected;
    expected = actual;
    return was_changed;
  }
};

template <MemoryScope Scope = MemoryScope::kDevice>
struct AtomicMutex {
  Atomic<int, Scope> __value;

  AtomicMutex(const AtomicMutex&) = delete;
  AtomicMutex& operator=(const AtomicMutex&) = delete;
  AtomicMutex& operator=(const AtomicMutex&) volatile = delete;

  WAVE_DEVICE_HOST explicit constexpr AtomicMutex() noexcept = default;
  WAVE_DEVICE_HOST constexpr explicit inline AtomicMutex(int value) noexcept
      : __value(value) {}

  WAVE_DEVICE_HOST void acquire() {
    constexpr int kPollingCount = 5;
    constexpr int kInitialBackoffStepNs = 10;

    int count = 0;
    int step_ns = kInitialBackoffStepNs + threadIdx.x % 32;
    for (;;) {
      int available = __value.template load<MemoryOrder::kAcquire>();
      while (available) {
        if (__value.template compare_exchange<MemoryOrder::kAcquire>(
                available, 0)) {
          return;
        }
      }
      if (count < kPollingCount) {
        count += 1;
        continue;
      }
#if defined(__CUDA_ARCH__)
      __nanosleep(step_ns);
      step_ns *= 2;
#endif
    }
  }

  WAVE_DEVICE_HOST void release() {
    __value.template store<MemoryOrder::kRelease>(1);
  }
};

} // namespace facebook::velox::wave
