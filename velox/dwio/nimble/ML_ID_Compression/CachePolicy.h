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
#pragma once

// Cache-state control for decode measurements.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>

#include <fstream>
#include <sstream>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#define ENCODINGS_BENCH_HAVE_CLFLUSH 1
#include <immintrin.h>
#else
#define ENCODINGS_BENCH_HAVE_CLFLUSH 0
#endif

namespace facebook::nimble::mlidc {

enum class CacheState {
  Hot,
  ColdPayload,
  ColdAll,
  ColdFileBuffered,  // posix_fadvise DONTNEED then normal pread
  ColdFileDirect,    // O_DIRECT bypass (caller opens file with O_DIRECT)
  ColdSystemDrop,    // /proc/sys/vm/drop_caches (root required, opt-in only)
};
enum class EvictMethod { Auto, Clflush, LlcThrash, None };

inline const char* cacheStateName(CacheState s) {
  switch (s) {
    case CacheState::Hot:
      return "hot";
    case CacheState::ColdPayload:
      return "cold-payload";
    case CacheState::ColdAll:
      return "cold-all";
    case CacheState::ColdFileBuffered:
      return "cold-file-buffered";
    case CacheState::ColdFileDirect:
      return "cold-file-direct";
    case CacheState::ColdSystemDrop:
      return "cold-system-drop";
  }
  return "hot";
}

inline const char* evictMethodName(EvictMethod m) {
  switch (m) {
    case EvictMethod::Auto:
      return "auto";
    case EvictMethod::Clflush:
      return "clflush";
    case EvictMethod::LlcThrash:
      return "llc-thrash";
    case EvictMethod::None:
      return "none";
  }
  return "auto";
}

struct CacheTopology {
  size_t l1dBytes{};
  size_t l2Bytes{};
  size_t llcBytes{};
  size_t lineBytes{64};

  static CacheTopology detect() {
    CacheTopology t;
    for (int idx = 0; idx < 16; ++idx) {
      const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index" +
          std::to_string(idx) + "/";
      const std::string levelStr = readSysfs(dir + "level");
      if (levelStr.empty())
        continue;
      const int level = std::atoi(levelStr.c_str());
      const std::string type = readSysfs(dir + "type");
      const size_t bytes = parseSize(readSysfs(dir + "size"));
      const size_t line = static_cast<size_t>(
          std::atoi(readSysfs(dir + "coherency_line_size").c_str()));
      if (line > 0)
        t.lineBytes = line;
      if (bytes == 0)
        continue;
      const bool isInstruction = (type.rfind("Instruction", 0) == 0);
      if (level == 1 && !isInstruction)
        t.l1dBytes = bytes;
      else if (level == 2 && !isInstruction)
        t.l2Bytes = bytes;
      if (!isInstruction && level >= 2 && bytes > t.llcBytes)
        t.llcBytes = bytes;
    }
    return t;
  }

  std::string describe() const {
    std::ostringstream os;
    os << "l1d=" << humanBytes(l1dBytes) << " l2=" << humanBytes(l2Bytes)
       << " llc=" << humanBytes(llcBytes) << " line=" << lineBytes << "B";
    return os.str();
  }

  static std::string humanBytes(size_t bytes) {
    std::ostringstream os;
    if (bytes >= (size_t{1} << 20) && bytes % (size_t{1} << 20) == 0)
      os << (bytes >> 20) << "MiB";
    else if (bytes >= (size_t{1} << 20))
      os << (static_cast<double>(bytes) / 1048576.0) << "MiB";
    else if (bytes >= 1024 && bytes % 1024 == 0)
      os << (bytes >> 10) << "KiB";
    else
      os << bytes << "B";
    return os.str();
  }

 private:
  static std::string readSysfs(const std::string& path) {
    std::ifstream in(path);
    if (!in)
      return {};
    std::string line;
    std::getline(in, line);
    return line;
  }

  static size_t parseSize(const std::string& s) {
    if (s.empty())
      return 0;
    size_t value = 0;
    size_t i = 0;
    for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i)
      value = value * 10 + static_cast<size_t>(s[i] - '0');
    if (i < s.size()) {
      if (s[i] == 'K' || s[i] == 'k')
        value *= 1024;
      else if (s[i] == 'M' || s[i] == 'm')
        value *= 1024 * 1024;
      else if (s[i] == 'G' || s[i] == 'g')
        value *= 1024 * 1024 * 1024;
    }
    return value;
  }
};

struct CachePolicy {
  CacheState state{CacheState::Hot};
  EvictMethod method{EvictMethod::Auto};
  bool activeWarm{false};
  size_t clflushMaxBytes{0};
  double thrashMultiple{2.0};
  bool strict{true};
};

struct EvictionTargets {
  std::span<const std::byte> payload;
  std::span<std::byte> sink;
  std::vector<std::span<const std::byte>> codecInternal;
  std::optional<int> fileFd; // fd for file-backed eviction
  std::optional<size_t> fileSize; // file size for fadvise
};

class CacheController {
 public:
  CacheController(CachePolicy policy, CacheTopology topo)
      : policy_(policy), topo_(topo) {
    requestedMethod_ = policy.method;
    if (topo_.lineBytes == 0)
      topo_.lineBytes = 64;
    if (policy_.clflushMaxBytes == 0)
      policy_.clflushMaxBytes =
          (topo_.llcBytes ? topo_.llcBytes : size_t{16} << 20) * 4;
    policy_.method = resolveMethod(policy_.method);
    if (policy_.method == EvictMethod::LlcThrash) {
      const double base = static_cast<double>(
          topo_.llcBytes ? topo_.llcBytes : size_t{16} << 20);
      const double mult =
          policy_.thrashMultiple > 1.0 ? policy_.thrashMultiple : 2.0;
      thrash_.assign(static_cast<size_t>(base * mult), std::byte{1});
    }
  }

  void enableSystemDrop() {
    systemDropEnabled_ = true;
  }

  void warm(const EvictionTargets& t) {
    const auto t0 = Clock::now();
    uint64_t acc = 0;
    acc += touch(t.payload);
    acc += touch(std::span<const std::byte>(t.sink.data(), t.sink.size()));
    for (const auto& b : t.codecInternal)
      acc += touch(b);
    volatileSink_ = acc;
    lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - t0)
                       .count();
  }

  void prepare(const EvictionTargets& t) {
    latchSizeDependentMethod(t.payload.size());
    if (policy_.state == CacheState::Hot) {
      if (policy_.activeWarm) {
        warm(t);
        return;
      }
      lastEvictNs_ = 0;
      return;
    }
    if (policy_.state == CacheState::ColdFileBuffered) {
      prepareColdFileBuffered(t);
      return;
    }
    if (policy_.state == CacheState::ColdFileDirect) {
      lastEvictNs_ = 0;
      return;
    }
    if (policy_.state == CacheState::ColdSystemDrop) {
      prepareColdSystemDrop();
      return;
    }
    const auto t0 = Clock::now();
    switch (policy_.method) {
      case EvictMethod::Clflush:
        flushTargets(t);
        break;
      case EvictMethod::LlcThrash:
        thrash();
        break;
      case EvictMethod::None:
        break;
      case EvictMethod::Auto:
        break;
    }
    lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - t0)
                       .count();
  }

  int64_t lastEvictNs() const {
    return lastEvictNs_;
  }
  const CachePolicy& effectivePolicy() const {
    return policy_;
  }

  std::string describe() const {
    std::string s = std::string(cacheStateName(policy_.state)) + "/" +
        evictMethodName(policy_.method);
    if (policy_.method == EvictMethod::LlcThrash) {
      s += "(llc=" + CacheTopology::humanBytes(topo_.llcBytes) +
          ",buf=" + CacheTopology::humanBytes(thrash_.size()) + ")";
    } else if (policy_.method == EvictMethod::Clflush) {
      s += "(line=" + std::to_string(topo_.lineBytes) + "B)";
    }
    return s;
  }

 private:
  using Clock = std::chrono::high_resolution_clock;

  EvictMethod resolveMethod(EvictMethod requested) const {
    if (policy_.state == CacheState::Hot)
      return EvictMethod::None;
    if (policy_.state == CacheState::ColdFileBuffered ||
        policy_.state == CacheState::ColdFileDirect ||
        policy_.state == CacheState::ColdSystemDrop)
      return EvictMethod::None;
    if (requested == EvictMethod::Auto) {
      if (policy_.state == CacheState::ColdPayload) {
#if ENCODINGS_BENCH_HAVE_CLFLUSH
        return EvictMethod::Clflush;
#else
        return EvictMethod::LlcThrash;
#endif
      }
      return EvictMethod::LlcThrash;
    }
#if !ENCODINGS_BENCH_HAVE_CLFLUSH
    if (requested == EvictMethod::Clflush) {
      if (policy_.strict)
        throw std::runtime_error(
            "CacheController: clflush requested but not x86");
      return EvictMethod::LlcThrash;
    }
#endif
    if (requested == EvictMethod::None && policy_.strict)
      throw std::runtime_error(
          std::string("CacheController: state '") +
          cacheStateName(policy_.state) + "' with evict-method 'none'");
    return requested;
  }

  void latchSizeDependentMethod(size_t payloadBytes) {
    if (sizeLatched_) {
      if (policy_.strict && requestedMethod_ == EvictMethod::Auto &&
          policy_.method == EvictMethod::Clflush &&
          payloadBytes > policy_.clflushMaxBytes)
        throw std::runtime_error(
            "payload grew past clflushMaxBytes after latch");
      return;
    }
    sizeLatched_ = true;
    if (requestedMethod_ == EvictMethod::Auto &&
        policy_.method == EvictMethod::Clflush &&
        payloadBytes > policy_.clflushMaxBytes) {
      policy_.method = EvictMethod::LlcThrash;
      if (thrash_.empty()) {
        const double base = static_cast<double>(
            topo_.llcBytes ? topo_.llcBytes : size_t{16} << 20);
        const double mult =
            policy_.thrashMultiple > 1.0 ? policy_.thrashMultiple : 2.0;
        thrash_.assign(static_cast<size_t>(base * mult), std::byte{1});
      }
    }
  }

  void prepareColdFileBuffered(const EvictionTargets& t) {
    if (!t.fileFd.has_value())
      throw std::runtime_error(
          "ColdFileBuffered requires fileFd in EvictionTargets");
    const auto t0 = Clock::now();
    posix_fadvise(
        *t.fileFd,
        0,
        static_cast<off_t>(t.fileSize.value_or(0)),
        POSIX_FADV_DONTNEED);
    lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - t0)
                       .count();
  }

  void prepareColdSystemDrop() {
    if (!systemDropEnabled_)
      throw std::runtime_error(
          "ColdSystemDrop requires explicit opt-in via enableSystemDrop()");
    fprintf(
        stderr,
        "WARNING: CacheController is dropping system page cache via "
        "/proc/sys/vm/drop_caches (requires root)\n");
    const auto t0 = Clock::now();
    std::ofstream f("/proc/sys/vm/drop_caches");
    if (!f)
      throw std::runtime_error(
          "Cannot write to /proc/sys/vm/drop_caches (requires root)");
    f << "3" << std::flush;
    lastEvictNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - t0)
                       .count();
  }

  void flushTargets(const EvictionTargets& t) {
    flushRange(t.payload);
    if (policy_.state == CacheState::ColdAll) {
      flushRange(std::span<const std::byte>(t.sink.data(), t.sink.size()));
      for (const auto& b : t.codecInternal)
        flushRange(b);
    }
#if ENCODINGS_BENCH_HAVE_CLFLUSH
    _mm_mfence();
#endif
  }

  void flushRange(std::span<const std::byte> r) {
#if ENCODINGS_BENCH_HAVE_CLFLUSH
    const std::byte* p = r.data();
    const std::byte* end = p + r.size();
    for (; p < end; p += topo_.lineBytes)
      _mm_clflush(p);
#else
    (void)r;
#endif
  }

  void thrash() {
    uint64_t acc = 0;
    const size_t stride = topo_.lineBytes;
    for (size_t i = 0; i < thrash_.size(); i += stride)
      acc += static_cast<uint64_t>(thrash_[i]);
    volatileSink_ = acc;
  }

  uint64_t touch(std::span<const std::byte> r) const {
    uint64_t acc = 0;
    const size_t stride = topo_.lineBytes;
    for (size_t i = 0; i < r.size(); i += stride)
      acc += static_cast<uint64_t>(r[i]);
    return acc;
  }

  CachePolicy policy_;
  CacheTopology topo_;
  EvictMethod requestedMethod_{EvictMethod::Auto};
  std::vector<std::byte> thrash_;
  volatile uint64_t volatileSink_{0};
  int64_t lastEvictNs_{0};
  bool sizeLatched_{false};
  bool systemDropEnabled_{false};
};

} // namespace facebook::nimble::mlidc
