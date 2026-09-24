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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

/// On-disk cache of encoded payloads, so a target is encoded once and every
/// later driver loads the bytes instead of re-encoding them.
///
/// A wrong hit is worse than no cache: every number downstream would be wrong
/// and mutually consistent, which is the hardest kind of error to notice. So
/// the key errs toward missing: anything not provably irrelevant is folded in,
/// and a spurious miss just costs the encode that would have happened anyway.
namespace facebook::nimble::mlidc {

/// 128 bits, as two independent FNV-1a streams with different bases and a
/// mixing step on the second.
///
/// Written out rather than taken from a library so the digest cannot change
/// underneath the cache and reshuffle every key at once.
struct Digest128 {
  uint64_t a{0xcbf29ce484222325ULL};
  uint64_t b{0x9ae16a3b2f90404fULL};

  void feed(const void* data, size_t size) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
      a = (a ^ p[i]) * 0x100000001b3ULL;
      b = (b ^ p[i]) * 0xff51afd7ed558ccdULL;
      b ^= b >> 29;
    }
  }

  /// Length-prefixed, so that feeding "ab" then "c" cannot collide with "a"
  /// then "bc".
  void feedString(const std::string& s) {
    const uint64_t n = s.size();
    feed(&n, sizeof(n));
    feed(s.data(), s.size());
  }

  std::string hex() const {
    char buf[33];
    std::snprintf(
        buf,
        sizeof(buf),
        "%016llx%016llx",
        static_cast<unsigned long long>(a),
        static_cast<unsigned long long>(b));
    return std::string(buf, 32);
  }
};

/// Whether a flag can change the encoded bytes.
///
/// Everything not listed here is folded into the key by default, so a flag
/// added later without being classified costs a miss, never a wrong hit.
inline bool flagAffectsEncoding(const std::string& name) {
  static const char* kMeasurementOnly[] = {
      "mlidc_iters",
      "mlidc_output_csv",
      "mlidc_output_manifest",
      "mlidc_dump_encoding",
      "mlidc_block_codec_iters",
      "mlidc_block_codec_probes",
      "mlidc_encode_cache_dir",
      "probes",
      "grid",
      "cache_state",
      "validate",
      "dry_run",
      "range_sizes",
      "range_offsets",
  };
  for (const char* m : kMeasurementOnly) {
    if (name == m) {
      return false;
    }
  }
  return true;
}

/// Root of the nimble tree, derived from this header's own compile-time path.
/// Following __FILE__ rather than a configured path keeps the fingerprint
/// correct in any checkout, including the worktrees this project measures from.
inline std::filesystem::path nimbleSourceRoot() {
  // .../velox/dwio/nimble/encodings/benchmarks/ml_id_compression/EncodeCache.h
  return std::filesystem::path(__FILE__)
      .parent_path() // ml_id_compression
      .parent_path() // benchmarks
      .parent_path() // encodings
      .parent_path(); // nimble
}

/// Every source file that could change what the encoder emits.
inline std::vector<std::filesystem::path> encoderSources() {
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  const auto root = nimbleSourceRoot();
  for (std::filesystem::recursive_directory_iterator it(root, ec), end;
       it != end && !ec;
       it.increment(ec)) {
    if (!it->is_regular_file(ec)) {
      continue;
    }
    const auto ext = it->path().extension().string();
    if (ext == ".h" || ext == ".cpp") {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

/// Fingerprint of the encoder build.
///
/// Hashing the executable would key every driver separately and defeat the
/// point of the cache, since each driver is a different binary. So the
/// fingerprint hashes the encoder's *sources* instead, shared across drivers;
/// it is sound only while the binary matches them, which cacheIsStale()
/// enforces.
inline const std::string& buildFingerprint() {
  static const std::string kFingerprint = [] {
    Digest128 d;
    const auto root = nimbleSourceRoot();
    size_t count = 0;
    for (const auto& f : encoderSources()) {
      std::ifstream in(f, std::ios::binary);
      if (!in) {
        continue;
      }
      d.feedString(std::filesystem::relative(f, root).string());
      std::vector<char> buf(size_t{1} << 16);
      while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto got = in.gcount();
        if (got > 0) {
          d.feed(buf.data(), static_cast<size_t>(got));
        }
      }
      ++count;
    }
    if (count == 0) {
      // Sources unreadable: key on something that cannot collide with a real
      // fingerprint rather than pretend two builds are the same.
      return std::string("NO-SOURCE-FINGERPRINT");
    }
    return d.hex();
  }();
  return kFingerprint;
}

/// Whether this binary predates the sources the fingerprint was taken from.
///
/// If a source is newer than the running executable, the binary holds an
/// older encoder than the key claims, so the cache turns itself off loudly
/// rather than silently serving stale results.
inline bool cacheIsStale() {
  static const bool kStale = [] {
    std::error_code ec;
    const auto exeTime = std::filesystem::last_write_time("/proc/self/exe", ec);
    if (ec) {
      return true;
    }
    for (const auto& f : encoderSources()) {
      const auto srcTime = std::filesystem::last_write_time(f, ec);
      if (ec) {
        continue;
      }
      if (srcTime > exeTime) {
        std::fprintf(
            stderr,
            "  [encode-cache] DISABLED: %s is newer than this binary. "
            "Rebuild before measuring.\n",
            f.string().c_str());
        return true;
      }
    }
    return false;
  }();
  return kStale;
}

/// Fingerprint of every flag that is not known to be measurement-only.
inline const std::string& flagsFingerprint() {
  static const std::string kFlags = [] {
    std::vector<gflags::CommandLineFlagInfo> all;
    gflags::GetAllFlags(&all);
    std::vector<std::string> parts;
    parts.reserve(all.size());
    for (const auto& f : all) {
      if (flagAffectsEncoding(f.name)) {
        parts.push_back(f.name + "=" + f.current_value);
      }
    }
    std::sort(parts.begin(), parts.end());
    Digest128 d;
    for (const auto& p : parts) {
      d.feedString(p);
    }
    return d.hex();
  }();
  return kFlags;
}

/// Which arm is being built is known in the driver loop and nowhere below it,
/// so the loop hands it over here rather than every encode signature growing an
/// argument it would only pass through.
struct CacheContext {
  std::string arm;
  bool valid{false};
};

inline CacheContext& cacheContext() {
  static thread_local CacheContext ctx;
  return ctx;
}

inline void setCacheContext(std::string arm) {
  auto& c = cacheContext();
  c.arm = std::move(arm);
  c.valid = true;
}

inline void clearCacheContext() {
  cacheContext().valid = false;
}

// The arm identity used for the cache key.
//
// realNestedSelection and option fields are folded in explicitly so a reader
// does not have to reconstruct them from the build fingerprint to trust the
// key.
inline std::string cacheArmIdentity(
    const Encoding::Options& options,
    bool realNestedSelection) {
  const auto& ctx = cacheContext();
  std::string id = ctx.valid ? ctx.arm : std::string("<unnamed>");
  id += realNestedSelection ? "|rn1" : "|rn0";
  id += "|t" + std::to_string(options.subIntSplitTransform);
  id += "|k" + std::to_string(options.subIntSplitKeySection);
  id += "|f" + std::to_string(options.subIntSplitForceApply ? 1 : 0);
  id += "|a" + std::to_string(options.subIntSplitAutoTransform ? 1 : 0);
  id += "|h" + std::to_string(options.subIntSplitAllowHuffman ? 1 : 0);
  id += "|d" + std::to_string(options.subIntSplitAllowDeltaBlock ? 1 : 0);
  id += "|n" + std::to_string(options.subIntSplitInNestedStreams ? 1 : 0);
  // A restricted inventory changes which encodings the planner may pick, so
  // it must be in the key or withdrawn and unrestricted arms would collide.
  id += "|q" +
      std::to_string(options.subIntSplitAllowedEncodings.empty() ? 0 : 1);
  // The decode weight and its access pattern change which encodings the split
  // planner picks, so they must be in the key.
  id += "|w" + std::to_string(options.subIntSplitDecodeWeight);
  id += "|p" +
      std::to_string(static_cast<int>(options.subIntSplitDecodeAccessPattern));
  // The read path selects which rate table prices decode.
  id += "|rp" +
      std::to_string(static_cast<int>(options.subIntSplitDecodeReadPath));
  // The size bound decides whether the weighted plan or the size-only one is
  // written.
  id += "|r" + std::to_string(options.subIntSplitMaxSizeRegression);
  // Admission decides whether nested streams may pick SubIntSplit.
  id += "|adm" + std::to_string(static_cast<int>(options.subIntSplitAdmission));
  // Whether an admitted stream still has to win on size changes which
  // encoding is written, so it must be in the key.
  if (options.subIntSplitAdmissionForces) {
    id += "|admf";
  }
  id += "|v" + std::to_string(options.useVarintRowCount ? 1 : 0);
  // deltaZigzagAnchorStride is deliberately absent: no encoding here reads it,
  // and the build fingerprint covers it if that ever changes.
  return id;
}

inline std::string cacheDir() {
  if (cacheIsStale()) {
    return {};
  }
  std::string value;
  gflags::GetCommandLineOption("mlidc_encode_cache_dir", &value);
  return value;
}

/// Builds the key for one encode.
///
/// The input values are hashed rather than the flags that produced them, so
/// hashing cannot be wrong about a flag nobody thought to include.
template <typename T>
std::string encodeCacheKey(
    const T* values,
    size_t count,
    const std::string& arm,
    EncodingType declaredType) {
  Digest128 d;
  d.feedString(std::string("nimble-encode-cache-v1"));
  d.feedString(arm);
  const uint64_t type = static_cast<uint64_t>(declaredType);
  d.feed(&type, sizeof(type));
  const uint64_t elementSize = sizeof(T);
  d.feed(&elementSize, sizeof(elementSize));
  const uint64_t rows = count;
  d.feed(&rows, sizeof(rows));
  d.feed(values, count * sizeof(T));
  d.feedString(buildFingerprint());
  d.feedString(flagsFingerprint());
  return d.hex();
}

inline std::string entryPath(const std::string& dir, const std::string& key) {
  return dir + "/enc_" + key + ".bin";
}

/// Loads a cached payload, or returns false.
///
/// The header repeats what the key already covers, so a hit is checked rather
/// than trusted: a mismatch shows up as a miss and a warning, not wrong bytes.
inline bool loadCached(
    const std::string& key,
    const std::string& arm,
    EncodingType declaredType,
    std::string& out) {
  const auto dir = cacheDir();
  if (dir.empty()) {
    return false;
  }
  std::ifstream in(entryPath(dir, key), std::ios::binary);
  if (!in) {
    return false;
  }
  std::string magic;
  std::string storedKey;
  std::string storedArm;
  std::string storedDigest;
  uint64_t storedType = 0;
  uint64_t storedSize = 0;
  in >> magic >> storedKey >> storedArm >> storedType >> storedSize >>
      storedDigest;
  in.get(); // the newline that ends the header
  if (!in || magic != "NIMBLE_ENCODE_CACHE_V1") {
    std::fprintf(
        stderr, "  [encode-cache] malformed entry ignored: %s\n", key.c_str());
    return false;
  }
  if (storedKey != key || storedArm != arm ||
      storedType != static_cast<uint64_t>(declaredType)) {
    std::fprintf(
        stderr,
        "  [encode-cache] entry disagrees with its key (arm '%s' vs '%s', "
        "type %llu vs %llu) -- ignored\n",
        storedArm.c_str(),
        arm.c_str(),
        static_cast<unsigned long long>(storedType),
        static_cast<unsigned long long>(declaredType));
    return false;
  }
  std::string payload(storedSize, '\0');
  in.read(payload.data(), static_cast<std::streamsize>(storedSize));
  if (static_cast<uint64_t>(in.gcount()) != storedSize) {
    std::fprintf(
        stderr, "  [encode-cache] short read ignored: %s\n", key.c_str());
    return false;
  }
  Digest128 d;
  d.feed(payload.data(), payload.size());
  if (d.hex() != storedDigest) {
    std::fprintf(
        stderr,
        "  [encode-cache] payload digest mismatch ignored: %s\n",
        key.c_str());
    return false;
  }
  out = std::move(payload);
  return true;
}

/// Stores a payload under `key`.
///
/// Written to a temporary and renamed, so a run killed mid-write cannot leave a
/// short file that a later run would read as a hit.
inline void storeCached(
    const std::string& key,
    const std::string& arm,
    EncodingType declaredType,
    const std::string& payload) {
  const auto dir = cacheDir();
  if (dir.empty()) {
    return;
  }
  Digest128 d;
  d.feed(payload.data(), payload.size());
  const auto finalPath = entryPath(dir, key);
  const auto tmpPath = finalPath + ".tmp";
  {
    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out) {
      return;
    }
    out << "NIMBLE_ENCODE_CACHE_V1 " << key << " " << arm << " "
        << static_cast<uint64_t>(declaredType) << " " << payload.size() << " "
        << d.hex() << "\n";
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!out) {
      return;
    }
  }
  std::rename(tmpPath.c_str(), finalPath.c_str());
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
