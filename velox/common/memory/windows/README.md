# Windows POSIX Memory Compatibility Layer

## Overview

This directory contains Windows implementations of POSIX memory management functions used throughout Velox. The compatibility layer provides drop-in replacements for `mmap`, `munmap`, `madvise`, and `aligned_alloc` that translate POSIX semantics to Windows APIs.

## Files

- **PosixMemoryCompat.h** - Header-only compatibility layer with Windows memory management functions
- **PosixMemoryCompat.cpp** - Static member initialization for the mapping registry
- **README.md** - This file

## Implementation Features

### 1. mmap/munmap with State Tracking

The Windows implementation provides:

- **Mapping registry**: Maintains a thread-safe registry of all active mappings with base addresses and sizes
- **Alignment handling**: Handles Windows allocation granularity (64KB) for file-backed mappings
- **File offset support**: Adjusts pointers for non-aligned file offsets
- **Anonymous and file-backed mappings**: Supports both `MAP_ANONYMOUS` and file descriptor-based mappings via `VirtualAlloc` and `MapViewOfFile`
- **Thread-safe**: All registry operations protected by mutex

### 2. MADV_DONTNEED Support

The `posix_madvise` implementation handles `MADV_DONTNEED` by:
- Using `VirtualFree` with `MEM_DECOMMIT` to release physical memory
- Maintaining address space reservation for future use
- Matching POSIX semantics where pages can be reused after decommit

### 3. Aligned Allocation

- Uses `_aligned_malloc` / `_aligned_free` pair for proper memory alignment
- Validates alignment requirements (power of 2, non-zero)
- Sets `errno` appropriately on failures

### 4. Error Handling

- Translates Windows error codes to POSIX errno values
- Provides detailed error context for debugging
- Matches POSIX return conventions (0 for success, -1 for failure)

## Usage

### In Velox Source Files

Include the compatibility layer for Windows builds:

```cpp
#ifndef _WIN32
#include <sys/mman.h>
#else
#include "velox/common/memory/windows/PosixMemoryCompat.h"
#endif
```

The header defines macros that make the Windows implementations transparent to existing code:

```cpp
#define mmap posix_mmap
#define munmap posix_munmap
#define madvise posix_madvise
#define aligned_alloc posix_aligned_alloc
```

Velox code using POSIX memory functions works without modification:

```cpp
void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
// ... use memory ...
munmap(ptr, size);
```

## Implementation Details

### Mapping Registry

The `MappingRegistry` class maintains a global registry of active mappings:

```cpp
struct MappingInfo {
  void* baseAddress;      // OS-returned base (for unmapping)
  size_t mappedSize;      // Total size including alignment
  size_t requestedSize;   // User-requested size
  bool isFileBacked;      // File vs anonymous
  bool isLargePage;       // Reserved for future large page support
};
```

**Why needed:** Windows `UnmapViewOfFile` and `VirtualFree` require the exact base pointer returned by the OS, but POSIX `mmap` may return an adjusted pointer (e.g., for file offset alignment).

### File-Backed Mapping Flow

1. Caller requests: `mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE, fd, 1024)`
2. Windows allocation granularity = 64KB, so offset 1024 must be adjusted:
   - Aligned offset = 0 (1024 rounded down to 64KB boundary)
   - Delta = 1024 bytes
3. Create file mapping: `CreateFileMappingW(fd, ...)`
4. Map view: `MapViewOfFile(..., offset=0, size=4096+1024)`
5. Return adjusted pointer: `base + 1024`
6. Registry stores: `{base, 4096+1024, 4096, true, false}`
7. On `munmap`: Look up base, call `UnmapViewOfFile(base)`

### MADV_DONTNEED Implementation

```cpp
case MADV_DONTNEED:
  VirtualFree(addr, length, MEM_DECOMMIT);  // Release physical memory
  // Address reservation kept - can be recommitted later
  break;
```

This matches POSIX semantics where:
- Physical memory is released
- Virtual address range remains valid
- Future access causes page fault and zero-fill (Windows) or undefined (POSIX)

## Testing Recommendations

1. **Alignment tests**: Verify file-backed mappings with various offsets
2. **Large mappings**: Test multi-GB allocations used by Velox
3. **Stress test**: Rapid alloc/free cycles to detect registry leaks
4. **Error paths**: Verify errno is set correctly on failures
5. **Thread safety**: Concurrent mmap/munmap from multiple threads

## Future Enhancements

- **Large page support**: Add `MAP_HUGETLB` handling via `VirtualAlloc` with `MEM_LARGE_PAGES`
- **NUMA awareness**: Support NUMA-specific allocations on Windows
- **Memory stats**: Track total mapped memory for diagnostics
- **PrefetchVirtualMemory**: Use Windows 8+ API for `MADV_WILLNEED`

## References

- [Windows Memory Management](https://docs.microsoft.com/en-us/windows/win32/memory/memory-management)
- [VirtualAlloc Documentation](https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)
- [CreateFileMapping Documentation](https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createfilemappinga)
- [POSIX mmap Specification](https://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html)
