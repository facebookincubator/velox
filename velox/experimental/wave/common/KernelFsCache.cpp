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
#include <glog/logging.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string_view>

namespace fs = std::filesystem;

namespace facebook::velox::wave {

namespace {

// Marks a file a compile is still writing. A stem carrying it is never taken
// for a cache entry, so a triple left behind by a process that died between
// writing its temporaries and renaming them is ignored rather than adopted.
constexpr std::string_view kTempPrefix = ".tmp.";

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

// Names an entry by its own text, so two processes compiling the same kernel
// choose the same file rather than racing for a shared counter. FNV-1a over
// 128 bits: wide enough that a collision between two distinct kernels is not
// worth planning for, and free of any dependency this target does not already
// have. Not std::hash, which is 64 bits and whose value is free to differ
// between builds -- a name has to mean the same thing to every process.
std::string contentDigest(std::string_view text) {
  __uint128_t hash = (static_cast<__uint128_t>(0x6c62272e07bb0142ULL) << 64) |
      0x62b821756295c58dULL;
  const __uint128_t prime =
      (static_cast<__uint128_t>(0x1000000ULL) << 64) | 0x13bULL;
  for (unsigned char byte : text) {
    hash ^= byte;
    hash *= prime;
  }
  return fmt::format(
      "{:016x}{:016x}",
      static_cast<uint64_t>(hash >> 64),
      static_cast<uint64_t>(hash));
}

// Renames 'from' onto 'to' if it exists. Rename within a directory is atomic,
// which is what lets a reader in another process see either the previous file
// or the whole new one, never a prefix of it. Returns false if the temporary
// is missing, which means the compiler wrote nothing to publish.
bool publishFile(std::string_view from, std::string_view to) {
  std::error_code error;
  if (!fs::exists(from, error)) {
    return false;
  }
  fs::rename(from, to, error);
  return !error;
}

} // namespace

KernelFsCache::KernelFsCache(const std::string& cacheDir)
    : cacheDir_(cacheDir) {}

std::string KernelFsCache::cuPath(std::string_view stem) const {
  return fmt::format("{}/{}.cu", cacheDir_, stem);
}

std::string KernelFsCache::cubinPath(std::string_view stem) const {
  return fmt::format("{}/{}.cubin", cacheDir_, stem);
}

std::string KernelFsCache::namesPath(std::string_view stem) const {
  return fmt::format("{}/{}.cubin.names", cacheDir_, stem);
}

std::string KernelFsCache::tempStem() {
  // The pid separates processes and the serial separates concurrent compiles
  // within one, so no two writers on a host ever share a temporary. The
  // kTempPrefix is what the scan in init() recognises them by; a process that
  // dies mid-publish leaves a complete-looking triple behind, and without the
  // prefix it would be picked up as an ordinary entry under a name no one can
  // account for.
  return fmt::format(
      "{}{}.{}", kTempPrefix, static_cast<int64_t>(getpid()), compileSerial_++);
}

bool KernelFsCache::entryComplete(std::string_view stem) const {
  // Non-empty rather than merely present: a writer that died, or ran out of
  // disk, leaves an empty file that would otherwise read as a complete entry
  // whose text matches nothing.
  //
  // The size has to be taken through an error_code that is then checked. The
  // non-throwing fs::file_size reports failure by returning uintmax_t(-1),
  // which would sail through a bare '> 0' -- so a file that another process
  // trims between the stat and the read, precisely the race this guards, would
  // be reported complete.
  auto nonEmpty = [](const std::string& path) {
    std::error_code error;
    auto size = fs::file_size(path, error);
    return !error && size > 0;
  };
  return nonEmpty(cuPath(stem)) && nonEmpty(cubinPath(stem)) &&
      nonEmpty(namesPath(stem));
}

void KernelFsCache::init() {
  if (initialized_) {
    return;
  }
  initialized_ = true;

  std::error_code error;
  if (!fs::exists(cacheDir_, error)) {
    fs::create_directories(cacheDir_, error);
    if (error) {
      // Not fatal: kernels still compile, they just are not cached. Said out
      // loud because the symptom otherwise is silent recompilation on a
      // directory that never fills up.
      LOG(WARNING) << "Wave kernel cache disabled, cannot create " << cacheDir_
                   << ": " << error.message();
    }
    return;
  }

  // Tolerate the directory changing underneath the walk. Another process
  // publishing an entry, or trimming one, can make a name the iterator has
  // already handed out disappear before it is opened; the throwing overloads
  // would turn that into an exception on whatever thread happens to be
  // compiling. A file that vanishes mid-scan is simply one this process does
  // not know about yet.
  fs::directory_iterator scan(cacheDir_, error);
  if (error) {
    return;
  }
  // Incremented through the error_code overload for the same reason: the
  // range-for form advances with the throwing one.
  for (; scan != fs::directory_iterator(); scan.increment(error)) {
    if (error) {
      break;
    }
    const auto& entry = *scan;
    if (entry.path().extension() != ".cu") {
      continue;
    }
    // The stem is taken as written. An entry from an earlier version is named
    // by an ordinal and one from this version by its digest; either is found
    // by the text of its .cu, so a directory does not have to be discarded
    // when the naming changes. The one stem that is not an entry is a
    // temporary a dead process left mid-publish.
    auto stem = entry.path().stem().string();
    if (std::string_view(stem).substr(0, kTempPrefix.size()) == kTempPrefix) {
      continue;
    }
    if (!entryComplete(stem)) {
      continue;
    }
    auto text = readFile(entry.path().string());
    if (text.empty()) {
      continue;
    }
    auto hash = std::hash<std::string>{}(text);
    hashToEntries_[hash].push_back({stem});
    ++numEntries_;
  }
}

std::unique_ptr<CompiledKernel> KernelFsCache::getKernel(
    const std::string& keyArg,
    KernelGenFunc genFunc) {
  // Namespace the persistent cache by a salt over the resolved headers, target
  // arch, and NVRTC flags. These affect the compiled CUBIN but are not part of
  // the generated kernel source, so without the salt a header/arch/flag change
  // would leave the key unchanged and serve a stale cubin. Prepending it to the
  // key means it participates in the hash, the stored .cu, the collision
  // compare, and the forwarded in-memory CompiledKernel cache uniformly.
  const std::string key =
      fmt::format("// wave-cache-salt:{:016x}\n", kernelCacheSalt()) + keyArg;
  auto hash = std::hash<std::string>{}(key);
  const std::string digest = contentDigest(key);
  auto assignedTemp = std::make_shared<std::string>();

  auto wrappedGen = [this,
                     keyCopy = key,
                     hash,
                     digest,
                     genFuncCopy = std::move(genFunc),
                     assignedTemp]() -> KernelSpec {
    std::unique_lock lock(mutex_);
    init();

    // Adopt an entry another process published after this one scanned. Without
    // this the scan is a snapshot taken once, and a long-lived process would
    // recompile everything its neighbours added. Cheap because the entry's
    // name follows from the text: this is a stat of one known path, not a
    // rescan of the directory.
    auto known = hashToEntries_.find(hash);
    if (known == hashToEntries_.end() && entryComplete(digest) &&
        readFile(cuPath(digest)) == keyCopy) {
      hashToEntries_[hash].push_back({digest});
      ++numEntries_;
    }

    auto it = hashToEntries_.find(hash);
    if (it != hashToEntries_.end()) {
      for (auto& entry : it->second) {
        auto text = readFile(cuPath(entry.stem));
        if (text != keyCopy) {
          continue;
        }
        if (!entryComplete(entry.stem)) {
          continue;
        }
        auto namesFilePath = namesPath(entry.stem);
        auto cubinFilePath = cubinPath(entry.stem);
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

    // Compile into a private temporary. Publishing is a rename once the file
    // is whole, so a concurrent reader never sees a partial cubin, and two
    // processes compiling this same kernel cannot write over each other
    // mid-write -- they each finish their own temporary and the last rename
    // wins, both files being compilations of the same source.
    auto temp = tempStem();
    *assignedTemp = temp;
    spec.cubinPath = cubinPath(temp);

    spec.postCompile = [this, temp, digest, hash, keyCopy](
                           KernelSpec& /*compiled*/, std::exception_ptr error) {
      std::unique_lock lock(mutex_);
      std::error_code removeError;
      auto removeTemps = [&]() {
        // Only ever this process's own temporaries. A file already under its
        // final name may have been published by another process compiling the
        // same kernel, and removing that is how one process destroys another's
        // entry.
        fs::remove(cuPath(temp), removeError);
        fs::remove(cubinPath(temp), removeError);
        fs::remove(namesPath(temp), removeError);
      };
      if (error) {
        removeTemps();
        return;
      }
      // A digest collision between two different kernels would otherwise have
      // one overwrite the other. Vanishingly unlikely, but the cost of being
      // wrong is serving the wrong machine code, so give the newcomer its own
      // name and let the exact text compare below sort them out on lookup.
      //
      // A stat that fails is not evidence that a name is free. Taking it as
      // free is the one way this walk can stop on a name another kernel owns:
      // the three renames below are not atomic as a group, so a reader in that
      // window would pair that kernel's source with this one's cubin. An
      // unreadable name therefore ends the publish -- the entry goes uncached
      // and recompiles next time, which costs work rather than correctness.
      std::string stem = digest;
      for (int32_t suffix = 1;; ++suffix) {
        std::error_code statError;
        if (!fs::exists(cuPath(stem), statError)) {
          if (statError) {
            removeTemps();
            return;
          }
          break;
        }
        if (readFile(cuPath(stem)) == keyCopy) {
          break;
        }
        stem = fmt::format("{}_{}", digest, suffix);
      }

      // Write the .cu before any rename. Of the steps left it is the only one
      // that realistically fails -- a full disk -- and doing it first means a
      // failure has published nothing and leaves nothing but temporaries to
      // clean up.
      auto tempCu = cuPath(temp);
      {
        std::ofstream out(tempCu);
        out << keyCopy;
        out.flush();
        if (!out.good()) {
          removeTemps();
          return;
        }
      }
      // The cubin and its names go first and the .cu last: the .cu is what
      // marks an entry present, so publishing it last means an entry is never
      // visible before the files it promises.
      if (!publishFile(cubinPath(temp), cubinPath(stem)) ||
          !publishFile(namesPath(temp), namesPath(stem)) ||
          !publishFile(tempCu, cuPath(stem))) {
        removeTemps();
        return;
      }
      hashToEntries_[hash].push_back({stem});
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
    // Only this process's own temporaries are removed. The published entry, if
    // one was reached, is named by content and may since have been adopted by
    // another process; deleting by a name this process picked is what used to
    // let one process delete another's files.
    if (!assignedTemp->empty()) {
      std::error_code removeError;
      fs::remove(cuPath(*assignedTemp), removeError);
      fs::remove(cubinPath(*assignedTemp), removeError);
      fs::remove(namesPath(*assignedTemp), removeError);
    }
    throw;
  }
}

} // namespace facebook::velox::wave
