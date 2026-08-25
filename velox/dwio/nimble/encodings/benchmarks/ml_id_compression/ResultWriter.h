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

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <folly/FileUtil.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/dwio/nimble/common/Exceptions.h"

// Result output for the ML ID benchmark drivers: the CSV writer and the JSON
// run manifest that records the machine and the flags a result set came from.
// Kept apart from BenchCommon.h because neither depends on the encodings.

DECLARE_string(mlidc_output_csv);
DECLARE_string(mlidc_output_manifest);
DECLARE_int32(mlidc_rows);
DECLARE_int32(mlidc_iters);
DECLARE_int64(mlidc_seed);
DECLARE_string(mlidc_file);
DECLARE_string(mlidc_dataset_name);
DECLARE_string(mlidc_substream_compression);
DECLARE_string(mlidc_outer_compression);
DECLARE_int32(mlidc_block_codec_iters);
DECLARE_string(mlidc_dtype);
DECLARE_int32(mlidc_block_codec_probes);
DECLARE_string(mlidc_datasets);

namespace facebook::nimble::mlidc {

// ---------------------------------------------------------------------------
// CsvResultWriter
// ---------------------------------------------------------------------------

class CsvResultWriter {
 public:
  CsvResultWriter(
      const std::string& path,
      std::vector<std::string> columns)
      : path_(path), columns_(std::move(columns)) {
    file_.open(path_, std::ios::out | std::ios::trunc);
    NIMBLE_CHECK(file_.is_open(), "Cannot open CSV output: " + path_);
    // Write header.
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) {
        file_ << ',';
      }
      file_ << columns_[i];
    }
    file_ << '\n';
  }

  void beginRow() {
    row_.clear();
  }

  void set(const std::string& col, const std::string& value) {
    // Minimal CSV quoting: wrap in quotes if value contains comma, quote, or
    // newline.
    if (value.find_first_of(",\"\n") != std::string::npos) {
      std::string quoted = "\"";
      for (char c : value) {
        if (c == '"') {
          quoted += "\"\"";
        } else {
          quoted += c;
        }
      }
      quoted += '"';
      row_[col] = std::move(quoted);
    } else {
      row_[col] = value;
    }
  }

  void set(const std::string& col, int64_t value) {
    row_[col] = std::to_string(value);
  }

  void set(const std::string& col, double value) {
    std::ostringstream oss;
    oss << value;
    row_[col] = oss.str();
  }

  void endRow() {
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) {
        file_ << ',';
      }
      auto it = row_.find(columns_[i]);
      if (it != row_.end()) {
        file_ << it->second;
      }
      // Missing column → empty (null in CSV).
    }
    file_ << '\n';
  }

  void flush() {
    file_.flush();
  }

 private:
  std::string path_;
  std::vector<std::string> columns_;
  std::ofstream file_;
  std::map<std::string, std::string> row_;
};

// ---------------------------------------------------------------------------
// RunManifest
// ---------------------------------------------------------------------------

namespace detail {

inline std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    return "";
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  // Strip trailing newline.
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

inline std::string cpuModel() {
  std::ifstream f("/proc/cpuinfo");
  if (!f.is_open()) {
    return "unknown";
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto colon = line.find(':');
      if (colon != std::string::npos) {
        auto s = line.substr(colon + 1);
        // ltrim
        s.erase(0, s.find_first_not_of(" \t"));
        return s;
      }
    }
  }
  return "unknown";
}

inline folly::dynamic cacheTopology() {
  auto topo = CacheTopology::detect();
  return folly::dynamic::object(
      "l1d_bytes", static_cast<int64_t>(topo.l1dBytes))(
      "l2_bytes", static_cast<int64_t>(topo.l2Bytes))(
      "llc_bytes", static_cast<int64_t>(topo.llcBytes))(
      "line_bytes", static_cast<int64_t>(topo.lineBytes));
}

inline std::string scalingGovernor() {
  return readFile(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

inline std::string hostname() {
  char buf[256] = {};
  if (::gethostname(buf, sizeof(buf)) == 0) {
    return buf;
  }
  return "unknown";
}

} // namespace detail

// Write a JSON manifest sidecar describing the current run environment.
inline void writeRunManifest(const std::string& path) {
  folly::dynamic manifest = folly::dynamic::object;

  manifest["hostname"] = detail::hostname();
  manifest["cpu_model"] = detail::cpuModel();
  manifest["cache_topology"] = detail::cacheTopology();
  manifest["scaling_governor"] = detail::scalingGovernor();
  manifest["compiler_version"] = __VERSION__;

#ifdef MLIDC_GIT_SHA
  manifest["build_sha"] = MLIDC_GIT_SHA;
#else
  manifest["build_sha"] = "unknown";
#endif

  // Reproduce key flags.
  // The compression flags belong here as much as the sizes: two runs that
  // differ only in compressor produce different numbers under the same
  // encoder names, so a result set is not interpretable without them.
  manifest["flags"] = folly::dynamic::object(
      "mlidc_rows", FLAGS_mlidc_rows)(
      "mlidc_iters", FLAGS_mlidc_iters)(
      "mlidc_seed", FLAGS_mlidc_seed)(
      "mlidc_file", FLAGS_mlidc_file)(
      "mlidc_dataset_name", FLAGS_mlidc_dataset_name)(
      "mlidc_substream_compression", FLAGS_mlidc_substream_compression)(
      "mlidc_outer_compression", FLAGS_mlidc_outer_compression)(
      "mlidc_block_codec_iters", FLAGS_mlidc_block_codec_iters)(
      "mlidc_dtype", FLAGS_mlidc_dtype);

  auto json = folly::toPrettyJson(manifest);
  if (folly::writeFile(json, path.c_str())) {
    LOG(INFO) << "Manifest written to " << path;
  } else {
    LOG(WARNING) << "Failed to write manifest to " << path;
  }
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
