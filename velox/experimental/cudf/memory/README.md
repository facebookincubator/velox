# Pinned host allocator

An opt-in process-wide allocator for AsyncDataCache and GPU spill storage. It
does not replace the execution allocator or automatically register with a worker.

## Contract

- Keep the allocator alive until cache shutdown and release of every allocation.
- Register the cache once before concurrent use.
- Allocation returns CPU-ready memory through CCCL allocate_sync.
- Complete all GPU accesses before freeing; retain cache pins through H2D reads.
  CCCL deallocate_sync does not wait for application streams.
- Byte allocation and page runs are supported. Contiguous page allocation and
  growth return failure. Base-class zero-fill and reallocation fallbacks work.
- Alignment padding and page rounding count toward the shared live budget.
- numAllocated reports charged pages, including both cache and spill allocations;
  totalUsedBytes reports requested payload bytes. retainedBytes reports CUDA
  backing, including idle reusable memory. There are no mmap allocations.
- unmap is a no-op: evicted capacity stays in the pinned pool for reuse.
- NUMA placement is supplied by the launcher; CUDA 12.9 requires a node ID.
- Tiny cache entries still bypass this allocator.

## Capacity and backing-size rounding

CUDA can round max_pool_size upward. On the laptop's CUDA 12.9 / driver
580.178.04, requested limits of 16/31/32 MiB allowed 32 MiB; 33/48/64 MiB allowed
64 MiB; 96 and 128 MiB held exactly. We accept approximately 32 MiB of implicit
backing-size rounding rather than imposing a 32 MiB minimum or multiple.
The configured capacity must be positive and page-aligned; the live allocation
budget, including page rounding and alignment padding, is enforced exactly.

Allow for this small backing-size difference in worker headroom. The observed
32 MiB quantum is not a portable CUDA guarantee; retainedBytes exposes the
actual value for deployment monitoring. Free backing remains available to both
cache and spill allocations without returning it to the OS.

## Tests and benchmark

The cuDF build targets are velox_cudf_pinned_memory_test and
velox_cudf_pinned_memory_benchmark. Debug enables partial-run failure injection.
Tests cover alignment, capacity contention, cache eviction/refill, a cache pin
held during a deliberately stalled GPU read, checksummed SSD-cache round trips,
and spill/restore across non-default streams.

The benchmark takes [iterations=1000] [cache_cycles=10]. It compares direct CCCL
against the adapter for cold growth, warm reuse, and four-thread contention;
then compares cache traffic with small versus large competing spill buffers
and measures H2D/D2H throughput. Cold pool measurements exclude one-time CUDA
context initialization; warm allocation timings include allocation and free.
The test and benchmark fixtures explicitly select NUMA node 0.

## Embedding

Construct the allocator before the cache, using capacity and NUMA placement
provided by the worker launcher:

```cpp
auto allocator = std::make_shared<PinnedMemoryAllocator>(
    PinnedMemoryAllocator::Options{capacityBytes, numaNode});
auto cache = cache::AsyncDataCache::create(allocator.get());

// Spill allocations share capacity with cache entries. The base allocator
// retries capacity failures through the registered cache's makeSpace().
void* spill = allocator->allocateBytes(spillBytes);
if (spill == nullptr) {
  // No reclaimable cache space remains: fail or apply the caller's policy.
}
// ...complete all GPU use of spill before freeing it...
allocator->freeBytes(spill, spillBytes);

// After users/background work have stopped and all cache pins are released:
cache->shutdown();
```

Production callers should retain spill allocations in their existing RAII
owners so cancellation and exceptions also release them after GPU completion.
No host-to-disk spill implementation or query-level spill policy is introduced.

On CUDA allocation failure with logical headroom, the adapter asks the cache to
shrink in progressively larger batches. Host-pool fragmentation can require
more reclamation than the incoming request size. Reclamation runs outside the
allocator mutex (cache eviction calls back into freeBytes), is bounded by one
capacity's worth of progress per attempt, and stops when no cache can be freed.
It neither synchronizes application streams nor changes core cache policy APIs.
