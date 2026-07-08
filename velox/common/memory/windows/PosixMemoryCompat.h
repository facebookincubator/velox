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

#ifndef _WIN32
#error "This header is only for Windows builds. Include <sys/mman.h> on POSIX systems."
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <io.h>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <errno.h>

namespace facebook::velox::memory::windows {

// POSIX mmap protection flags
#ifndef PROT_NONE
#define PROT_NONE 0x0
#endif
#ifndef PROT_READ
#define PROT_READ 0x1
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 0x2
#endif
#ifndef PROT_EXEC
#define PROT_EXEC 0x4
#endif

// POSIX mmap flags
#ifndef MAP_SHARED
#define MAP_SHARED 0x01
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x02
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x20
#endif
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void*)-1)
#endif

// POSIX madvise flags
#ifndef MADV_NORMAL
#define MADV_NORMAL 0
#endif
#ifndef MADV_RANDOM
#define MADV_RANDOM 1
#endif
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 2
#endif
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif

/**
 * Tracking structure for memory mappings to support proper unmapping.
 * Windows requires the exact base address and size for unmapping operations.
 */
struct MappingInfo {
  void* baseAddress;      // Original base from VirtualAlloc/MapViewOfFile
  size_t mappedSize;      // Total size including alignment adjustments
  size_t requestedSize;   // Size requested by caller
  bool isFileBacked;      // File-backed vs anonymous
  bool isLargePage;       // Uses large pages (if available)
};

/**
 * Global registry for tracking active mappings.
 * Required because munmap must use the exact base pointer returned by the OS.
 * 
 * Uses Meyers Singleton pattern to avoid static initialization order fiasco.
 */
class MappingRegistry {
 private:
  // Meyers Singleton: initialized on first use, thread-safe in C++11+
  static std::mutex& getMutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<void*, MappingInfo>& getMappings() {
    static std::unordered_map<void*, MappingInfo> mappings;
    return mappings;
  }

  struct SystemInfo {
    DWORD allocationGranularity;
    DWORD pageSize;
    
    SystemInfo() {
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      allocationGranularity = si.dwAllocationGranularity;
      pageSize = si.dwPageSize;
    }
  };

  static const SystemInfo& getSystemInfo() {
    static SystemInfo info;
    return info;
  }

 public:
  static void registerMapping(void* ptr, const MappingInfo& info) {
    std::lock_guard<std::mutex> lock(getMutex());
    getMappings()[ptr] = info;
  }

  static bool unregisterMapping(void* ptr, MappingInfo& info) {
    std::lock_guard<std::mutex> lock(getMutex());
    auto& mappings = getMappings();
    auto it = mappings.find(ptr);
    if (it == mappings.end()) {
      return false;
    }
    info = it->second;
    mappings.erase(it);
    return true;
  }

  static DWORD getAllocationGranularity() {
    return getSystemInfo().allocationGranularity;
  }

  static DWORD getPageSize() {
    return getSystemInfo().pageSize;
  }
};

/**
 * Translate errno to Windows error code for better diagnostics.
 */
inline void setErrnoFromWinError() {
  DWORD err = GetLastError();
  switch (err) {
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
      errno = ENOMEM;
      break;
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_FUNCTION:
      errno = EINVAL;
      break;
    case ERROR_ACCESS_DENIED:
      errno = EACCES;
      break;
    case ERROR_INVALID_HANDLE:
      errno = EBADF;
      break;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      errno = ENOENT;
      break;
    default:
      errno = EINVAL;
      break;
  }
}

/**
 * Translate POSIX protection flags to Windows page protection.
 */
inline DWORD protToPageProtect(int prot) {
  if (prot == PROT_NONE) {
    return PAGE_NOACCESS;
  }
  if (prot & PROT_EXEC) {
    if (prot & PROT_WRITE) {
      return PAGE_EXECUTE_READWRITE;
    } else if (prot & PROT_READ) {
      return PAGE_EXECUTE_READ;
    } else {
      return PAGE_EXECUTE;
    }
  }
  if (prot & PROT_WRITE) {
    return PAGE_READWRITE;
  }
  if (prot & PROT_READ) {
    return PAGE_READONLY;
  }
  return PAGE_NOACCESS;
}

/**
 * Translate POSIX protection flags to Windows file mapping access.
 */
inline DWORD protToFileMapAccess(int prot) {
  DWORD access = 0;
  if (prot & PROT_READ) {
    access |= FILE_MAP_READ;
  }
  if (prot & PROT_WRITE) {
    access |= FILE_MAP_WRITE;
  }
  if (prot & PROT_EXEC) {
    access |= FILE_MAP_EXECUTE;
  }
  return access ? access : FILE_MAP_READ;
}

/**
 * POSIX-compatible mmap implementation for Windows.
 * 
 * Supports:
 * - Anonymous mappings (MAP_ANONYMOUS or fd == -1)
 * - File-backed mappings (fd >= 0)
 * - Proper alignment for file offsets (Windows allocation granularity)
 * - Tracking for correct munmap behavior
 * 
 * Returns: pointer to mapped region on success, MAP_FAILED on failure
 */
inline void* posix_mmap(
    void* addr,
    size_t length,
    int prot,
    int flags,
    int fd,
    off_t offset) {
  if (length == 0) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  // Anonymous mapping (most common case for Velox)
  if ((flags & MAP_ANONYMOUS) || fd == -1) {
    DWORD protect = protToPageProtect(prot);
    
    // VirtualAlloc on Windows: addr is a hint, can be NULL
    void* base = VirtualAlloc(
        addr,
        length,
        MEM_RESERVE | MEM_COMMIT,
        protect);
    
    if (!base) {
      setErrnoFromWinError();
      return MAP_FAILED;
    }

    // Register mapping for proper cleanup
    MappingInfo info;
    info.baseAddress = base;
    info.mappedSize = length;
    info.requestedSize = length;
    info.isFileBacked = false;
    info.isLargePage = false;

    MappingRegistry::registerMapping(base, info);
    return base;
  }

  // File-backed mapping
  intptr_t osfHandle = _get_osfhandle(fd);
  if (osfHandle == -1) {
    errno = EBADF;
    return MAP_FAILED;
  }
  HANDLE fileHandle = reinterpret_cast<HANDLE>(osfHandle);

  // Create file mapping object
  DWORD pageProtect = protToPageProtect(prot);
  HANDLE mappingHandle = CreateFileMappingW(
      fileHandle,
      nullptr,
      pageProtect,
      0, 0, // size: 0,0 means entire file
      nullptr);
  
  if (!mappingHandle) {
    setErrnoFromWinError();
    return MAP_FAILED;
  }

  // MapViewOfFile requires offset aligned to allocation granularity
  DWORD granularity = MappingRegistry::getAllocationGranularity();
  uint64_t uoffset = static_cast<uint64_t>(offset);
  uint64_t alignedOffset = (uoffset / granularity) * granularity;
  size_t delta = static_cast<size_t>(uoffset - alignedOffset);
  size_t viewSize = length + delta;

  DWORD offsetLow = static_cast<DWORD>(alignedOffset & 0xFFFFFFFFULL);
  DWORD offsetHigh = static_cast<DWORD>((alignedOffset >> 32) & 0xFFFFFFFFULL);
  DWORD access = protToFileMapAccess(prot);

  void* base = MapViewOfFile(
      mappingHandle,
      access,
      offsetHigh,
      offsetLow,
      viewSize);
  
  CloseHandle(mappingHandle); // View keeps reference, safe to close

  if (!base) {
    setErrnoFromWinError();
    return MAP_FAILED;
  }

  // Return adjusted pointer (base + delta to honor original offset)
  void* userPtr = static_cast<char*>(base) + delta;

  // Register with original base for proper cleanup
  MappingInfo info;
  info.baseAddress = base;
  info.mappedSize = viewSize;
  info.requestedSize = length;
  info.isFileBacked = true;
  info.isLargePage = false;

  MappingRegistry::registerMapping(userPtr, info);
  return userPtr;
}

/**
 * POSIX-compatible munmap implementation for Windows.
 * 
 * Unmaps memory previously mapped with posix_mmap.
 * Must be called with the exact pointer returned by posix_mmap.
 * 
 * Returns: 0 on success, -1 on failure (sets errno)
 */
inline int posix_munmap(void* addr, size_t length) {
  if (!addr || addr == MAP_FAILED) {
    errno = EINVAL;
    return -1;
  }

  MappingInfo info;
  if (!MappingRegistry::unregisterMapping(addr, info)) {
    // Not found in registry - might be external allocation
    // Try both unmap methods as fallback
    if (UnmapViewOfFile(addr)) {
      return 0;
    }
    if (VirtualFree(addr, 0, MEM_RELEASE)) {
      return 0;
    }
    errno = EINVAL;
    return -1;
  }

  // Use tracked base address for proper cleanup
  BOOL success;
  if (info.isFileBacked) {
    success = UnmapViewOfFile(info.baseAddress);
  } else {
    success = VirtualFree(info.baseAddress, 0, MEM_RELEASE);
  }

  if (!success) {
    setErrnoFromWinError();
    return -1;
  }

  return 0;
}

/**
 * POSIX-compatible madvise implementation for Windows.
 * 
 * Windows doesn't have direct equivalents for all madvise operations.
 * We implement the most critical ones for Velox:
 * - MADV_DONTNEED: Decommit pages (VirtualFree with MEM_DECOMMIT)
 * - MADV_WILLNEED: Prefetch hint (VirtualAlloc touch or no-op)
 * - Others: No-op (return success)
 * 
 * Returns: 0 on success, -1 on failure
 */
inline int posix_madvise(void* addr, size_t length, int advice) {
  if (!addr || length == 0) {
    errno = EINVAL;
    return -1;
  }

  switch (advice) {
    case MADV_DONTNEED: {
      // Use MEM_RESET instead of MEM_DECOMMIT to match Linux MADV_DONTNEED
      // semantics. MEM_DECOMMIT makes pages inaccessible (causes SEH 0xc0000005
      // on re-access), while MEM_RESET tells the OS the contents are no longer
      // needed but keeps pages committed and accessible.
      void* result = VirtualAlloc(addr, length, MEM_RESET, PAGE_READWRITE);
      if (!result) {
        setErrnoFromWinError();
        return -1;
      }
      return 0;
    }

    case MADV_WILLNEED: {
      // Prefetch hint - touch pages to bring into working set
      // Alternative: use PrefetchVirtualMemory on Windows 8+ (not implemented here)
      // For now, no-op is acceptable
      return 0;
    }

    case MADV_NORMAL:
    case MADV_RANDOM:
    case MADV_SEQUENTIAL:
      // Access pattern hints - no Windows equivalent, safe to ignore
      return 0;

    default:
      // Unknown advice - return success (POSIX allows this)
      return 0;
  }
}

/**
 * Registry for tracking aligned allocations to prevent heap corruption.
 * 
 * Windows requires aligned allocations (_aligned_malloc) to be freed with
 * _aligned_free, not regular free(). Since code uses free() universally,
 * we track ALL aligned allocations and intercept deallocation calls.
 * 
 * CRITICAL: This registry tracks allocations from:
 * - posix_aligned_alloc (via aligned_alloc macro)
 * - posix_memalign
 * All must be freed via the smart wrapper functions.
 */
class AlignedAllocationRegistry {
 private:
  static std::mutex& getMutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<void*, size_t>& getAllocations() {
    static std::unordered_map<void*, size_t> allocations;
    return allocations;
  }

 public:
  static void registerAllocation(void* ptr, size_t size) {
    if (ptr) {
      std::lock_guard<std::mutex> lock(getMutex());
      getAllocations()[ptr] = size;
    }
  }

  static bool isAlignedAllocation(void* ptr) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(getMutex());
    return getAllocations().find(ptr) != getAllocations().end();
  }

  static bool unregisterAllocation(void* ptr) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(getMutex());
    auto it = getAllocations().find(ptr);
    if (it != getAllocations().end()) {
      getAllocations().erase(it);
      return true;
    }
    return false;
  }

  static size_t getSize(void* ptr) {
    if (!ptr) return 0;
    std::lock_guard<std::mutex> lock(getMutex());
    auto it = getAllocations().find(ptr);
    return (it != getAllocations().end()) ? it->second : 0;
  }
};

/**
 * POSIX posix_memalign implementation for Windows.
 * 
 * Allocates size bytes aligned to alignment boundary.
 * Returns 0 on success with *memptr set to allocated memory.
 * Returns error code on failure (EINVAL or ENOMEM).
 * 
 * CRITICAL: Memory allocated here MUST be freed with posix_free(),
 * NOT regular free(), to prevent heap corruption.
 */
inline int posix_memalign(void** memptr, size_t alignment, size_t size) {
  if (!memptr) {
    return EINVAL;
  }

  // POSIX requires alignment to be:
  // - a power of 2
  // - a multiple of sizeof(void*)
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      (alignment % sizeof(void*)) != 0) {
    return EINVAL;
  }

  void* ptr = _aligned_malloc(size, alignment);
  if (!ptr) {
    return ENOMEM;
  }

  // Register this as an aligned allocation
  AlignedAllocationRegistry::registerAllocation(ptr, size);
  *memptr = ptr;
  return 0;
}

/**
 * Aligned allocation compatible with C11 aligned_alloc semantics.
 * Uses _aligned_malloc on Windows and registers the allocation for
 * proper cleanup via the wrapper free function.
 * 
 * CRITICAL: Memory allocated here will be tracked and automatically
 * freed with _aligned_free when posix_free() is called.
 */
inline void* posix_aligned_alloc(size_t alignment, size_t size) {
  // C11 aligned_alloc requires size to be a multiple of alignment
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    // alignment must be power of 2
    errno = EINVAL;
    return nullptr;
  }

  void* ptr = _aligned_malloc(size, alignment);
  if (!ptr) {
    errno = ENOMEM;
    return nullptr;
  }

  // Register this as an aligned allocation
  AlignedAllocationRegistry::registerAllocation(ptr, size);
  return ptr;
}

/**
 * Smart malloc wrapper that uses regular malloc but can be freed by posix_free.
 * Ensures consistency with the deallocation strategy.
 */
inline void* posix_malloc(size_t size) {
  return ::malloc(size);
}

/**
 * Smart calloc wrapper that uses regular calloc but can be freed by posix_free.
 * Ensures consistency with the deallocation strategy.
 */
inline void* posix_calloc(size_t num, size_t size) {
  return ::calloc(num, size);
}

/**
 * Smart realloc wrapper that handles both regular and aligned allocations.
 * 
 * CRITICAL BEHAVIOR:
 * - If ptr is nullptr, acts like malloc
 * - If ptr is an aligned allocation, this is UNSAFE (realloc doesn't support alignment)
 * - For safety, we detect aligned allocations and fail
 */
inline void* posix_realloc(void* ptr, size_t new_size) {
  if (!ptr) {
    // Acts like malloc
    return ::realloc(nullptr, new_size);
  }

  // Check if this is an aligned allocation
  if (AlignedAllocationRegistry::isAlignedAllocation(ptr)) {
    // Cannot safely realloc aligned memory - realloc doesn't preserve alignment
    // This is a programming error - aligned memory shouldn't be reallocated
    errno = EINVAL;
    return nullptr;
  }

  // Regular realloc
  return ::realloc(ptr, new_size);
}

/**
 * Smart free function that handles both regular and aligned allocations.
 * 
 * This function MUST be used instead of regular free() on Windows to prevent
 * heap corruption from freeing aligned allocations incorrectly.
 * 
 * Checks the registry to determine if this is an aligned allocation:
 * - If aligned: uses _aligned_free()
 * - If regular: uses free()
 */
inline void posix_free(void* ptr) {
  if (!ptr) {
    return;
  }

  if (AlignedAllocationRegistry::unregisterAllocation(ptr)) {
    // This was an aligned allocation - must use _aligned_free
    _aligned_free(ptr);
  } else {
    // Regular allocation - use standard free
    ::free(ptr);
  }
}

/**
 * Legacy function - prefer using posix_free() for all deallocations.
 * Kept for explicit aligned_free calls in existing code.
 */
inline void posix_aligned_free(void* ptr) {
  if (ptr) {
    AlignedAllocationRegistry::unregisterAllocation(ptr);
    _aligned_free(ptr);
  }
}

} // namespace facebook::velox::memory::windows

// Global namespace aliases for transparent usage in existing code
using facebook::velox::memory::windows::posix_mmap;
using facebook::velox::memory::windows::posix_munmap;
using facebook::velox::memory::windows::posix_madvise;
using facebook::velox::memory::windows::posix_memalign;
using facebook::velox::memory::windows::posix_aligned_alloc;
using facebook::velox::memory::windows::posix_aligned_free;
using facebook::velox::memory::windows::posix_malloc;
using facebook::velox::memory::windows::posix_calloc;
using facebook::velox::memory::windows::posix_realloc;
using facebook::velox::memory::windows::posix_free;

// Macro definitions for mmap/madvise (safe - these are POSIX-specific)
#define mmap posix_mmap
#define munmap posix_munmap
#define madvise posix_madvise

// NOTE: We do NOT override aligned_alloc with a macro because:
// - We cannot also override free() (breaks member functions, third-party libs)
// - Any code calling aligned_alloc() would get tracked, but free() would crash
// - Files that need aligned allocation must explicitly call ::posix_aligned_alloc()
//   AND ::posix_free() (MallocAllocator.cpp, MmapAllocator.cpp)
//
// Other files (RawVector.h, tests) must handle aligned memory themselves.

// NOTE: posix_memalign does NOT need a macro because:
// 1. It's already available via "using" declaration above
// 2. Unlike malloc/free, it's called explicitly by name (not generic pattern)
// 3. Code uses it as: int err = posix_memalign(&ptr, align, size);
// 4. On Windows, there's no native posix_memalign to shadow
//
// If code needs it, call: ::posix_memalign() or posix_memalign() directly.

// CRITICAL DECISION: We CANNOT use macros to override malloc/calloc/free/realloc
//
// Reasons:
// 1. Macros replace ALL occurrences, including member function names
//    Example: void SizeClass::free(...) becomes void SizeClass::posix_free(...)
// 2. Third-party libraries (Arrow, Azure SDK, Folly) would be affected
// 3. #ifndef guards don't work - they check for macros, not functions
// 4. Cannot selectively override aligned_alloc without also overriding free()
//    If we #define aligned_alloc posix_aligned_alloc, then ANY code calling
//    aligned_alloc() gets tracked, but if that code calls free(), it crashes.
//
// ROOT CAUSE OF HEAP CORRUPTION (Fixed November 20, 2025):
// - RawVector.h called aligned_alloc() → (via macro) → posix_aligned_alloc()
// - posix_aligned_alloc() used _aligned_malloc and registered the allocation
// - RawVector.h::freeData() called ::free() on the pointer
// - Windows ::free() cannot handle _aligned_malloc memory → CRASH
//
// SOLUTION: No aligned_alloc macro. Files must handle aligned memory explicitly:
//
// A. Velox allocator files (use tracking registry):
//    - velox/common/memory/MallocAllocator.cpp: ::posix_aligned_alloc + ::posix_free
//    - velox/common/memory/MmapAllocator.cpp: ::posix_aligned_alloc + ::posix_free
//
// B. Other files (handle directly without tracking):
//    - velox/common/base/RawVector.h: _aligned_malloc + _aligned_free (Windows)
//    - velox/common/base/tests/Memcpy.cpp: aligned_alloc + _aligned_free (Windows)
