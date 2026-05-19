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

#include "velox/experimental/wave/common/KernelFsCache.h"

#include <fmt/format.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>

namespace fs = std::filesystem;

namespace facebook::velox::wave {

namespace {

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) {
    return {};
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::string> readNames(const std::string& path) {
  std::vector<std::string> result;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      result.push_back(line);
    }
  }
  return result;
}

} // namespace

KernelFsCache::KernelFsCache(const std::string& cacheDir)
    : cacheDir_(cacheDir) {}

std::string KernelFsCache::cuPath(int32_t ordinal) const {
  return fmt::format("{}/{}.cu", cacheDir_, ordinal);
}

std::string KernelFsCache::cubinPath(int32_t ordinal) const {
  return fmt::format("{}/{}.cubin", cacheDir_, ordinal);
}

std::string KernelFsCache::namesPath(int32_t ordinal) const {
  return fmt::format("{}/{}.cubin.names", cacheDir_, ordinal);
}

void KernelFsCache::init() {
  if (initialized_) {
    return;
  }
  initialized_ = true;

  if (!fs::exists(cacheDir_)) {
    fs::create_directories(cacheDir_);
    return;
  }

  for (auto& entry : fs::directory_iterator(cacheDir_)) {
    if (entry.path().extension() != ".cu") {
      continue;
    }
    auto stem = entry.path().stem().string();
    int32_t ordinal;
    try {
      ordinal = std::stoi(stem);
    } catch (...) {
      continue;
    }
    auto cubinFile = cubinPath(ordinal);
    auto namesFile = namesPath(ordinal);
    if (!fs::exists(cubinFile) || fs::file_size(cubinFile) == 0 ||
        !fs::exists(namesFile)) {
      continue;
    }
    auto text = readFile(entry.path().string());
    if (text.empty()) {
      continue;
    }
    auto hash = std::hash<std::string>{}(text);
    hashToEntries_[hash].push_back({ordinal});
    ++numEntries_;
    if (ordinal >= nextOrdinal_) {
      nextOrdinal_ = ordinal + 1;
    }
  }
}

std::unique_ptr<CompiledKernel> KernelFsCache::getKernel(
    const std::string& key,
    KernelGenFunc genFunc) {
  auto hash = std::hash<std::string>{}(key);
  auto assignedOrdinal = std::make_shared<std::atomic<int32_t>>(-1);

  auto wrappedGen = [this,
                     keyCopy = key,
                     hash,
                     genFuncCopy = std::move(genFunc),
                     assignedOrdinal]() -> KernelSpec {
    std::unique_lock lock(mutex_);
    init();

    auto it = hashToEntries_.find(hash);
    if (it != hashToEntries_.end()) {
      for (auto& entry : it->second) {
        auto text = readFile(cuPath(entry.ordinal));
        if (text != keyCopy) {
          continue;
        }
        auto cubinFilePath = cubinPath(entry.ordinal);
        auto namesFilePath = namesPath(entry.ordinal);
        if (!fs::exists(cubinFilePath) || fs::file_size(cubinFilePath) == 0 ||
            !fs::exists(namesFilePath)) {
          continue;
        }
        auto lowered = readNames(namesFilePath);
        lock.unlock();
        ++hits_;

        KernelSpec loadSpec;
        loadSpec.loweredNames = std::move(lowered);
        loadSpec.fromCubinPath = std::move(cubinFilePath);
        return loadSpec;
      }
    }

    lock.unlock();
    auto spec = genFuncCopy();
    lock.lock();

    auto ordinal = nextOrdinal_++;
    assignedOrdinal->store(ordinal);
    spec.cubinPath = cubinPath(ordinal);

    spec.postCompile = [this, ordinal, hash, keyCopy](
                           KernelSpec& /*compiled*/, std::exception_ptr error) {
      std::unique_lock lock(mutex_);
      if (error) {
        std::error_code removeError;
        fs::remove(cubinPath(ordinal), removeError);
        fs::remove(cubinPath(ordinal) + ".names", removeError);
        return;
      }
      {
        std::ofstream out(cuPath(ordinal));
        out << keyCopy;
      }
      hashToEntries_[hash].push_back({ordinal});
      ++numEntries_;
    };

    ++misses_;
    lock.unlock();

    return spec;
  };

  try {
    return CompiledKernel::getKernel(key, std::move(wrappedGen));
  } catch (...) {
    std::unique_lock lock(mutex_);
    auto ordinal = assignedOrdinal->load();
    if (ordinal >= 0) {
      std::error_code removeError;
      fs::remove(cuPath(ordinal), removeError);
      fs::remove(cubinPath(ordinal), removeError);
      fs::remove(namesPath(ordinal), removeError);
      auto it = hashToEntries_.find(hash);
      if (it != hashToEntries_.end()) {
        auto& entries = it->second;
        entries.erase(
            std::remove_if(
                entries.begin(),
                entries.end(),
                [&](const CacheEntry& entry) {
                  return entry.ordinal == ordinal;
                }),
            entries.end());
        if (entries.empty()) {
          hashToEntries_.erase(it);
        }
      }
    }
    throw;
  }
}

} // namespace facebook::velox::wave
