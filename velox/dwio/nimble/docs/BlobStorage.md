# Nimble Blob Storage

This document describes how Nimble stores large `VARBINARY` and `VARCHAR` values
out of line, so that scans pay for a payload only when they actually materialize
it.

## Motivation

Nimble's unit of I/O is a chunk of a column stream. That is efficient for small
uniform values and wrong for large ones. Column projection lets a scan skip a
column entirely, but once the column is projected there is no way to want a
single row: reading one value means reading and decoding the chunk that contains
it. Encoding selection also does no useful work on multi-megabyte opaque
payloads, and string encoding stores `uint32` lengths plus concatenated bytes,
which does not suit payload-heavy data.

Blob storage separates the reference from the payload. The column stream keeps
one small integer per row; the bytes live in their own region and are fetched on
demand.

```
            inline                                  blob-backed

   payload column stream                  payload column stream   blob region
  +---------------------+                +-----------------+   +--------------+
  |  ####  ####  ####   |                |   0   1   2     |-->| ####  ####   |
  |    (megabytes)      |                |    (bytes)      |   | (megabytes)  |
  +---------------------+                +-----------------+   +--------------+
   one stream,                            small id stream,      random access,
   all or nothing                         scans cheaply         read per value
```

## Logical and physical model

A blob-backed column stays an ordinary logical column. Its schema type is
unchanged and readers see `VARBINARY` or `VARCHAR`. Only the physical
representation differs: the value stream holds `Int64` blob ids, and the stream
reports `Int64` as its physical scalar kind so the ids are encoded with integer
encodings rather than string encodings.

Because blob ids are assigned in row order and start at zero, the id stream is a
dense monotonic sequence and compresses to almost nothing.

Nulls consume no blob id and no entry; they use the stream's existing nullable
path. An empty value is a real blob id that resolves to a zero-length payload.

A `nimble.blob_store.id` schema attribute on the column records which blob store
backs it. That attribute is how the reader decides to build a blob-aware field
reader.

## Blob stores and blob ids

A **blob store** is a tablet-local id namespace, one per blob-backed column. Ids
restart at zero in every store, so a blob id alone is ambiguous and the identity
of a payload is the pair `(blobStoreId, blobId)`.

```
  column "payload"    -> store 0 -> blob ids  0, 1, 2, 3, ...
  column "thumbnail"  -> store 1 -> blob ids  0, 1, 2, 3, ...
```

Keeping ids per store rather than per file lets each column's id stream stay
dense from zero. The row stores only the blob id; the reader supplies the store
id, which it already knows from the schema attribute.

Blob ids are tablet-local and are not durable object identifiers. They carry no
information about where the bytes live, which leaves room to relocate payloads
without changing the row format.

## File layout

Payload bytes are written at the end of the stripe that produced them, so a
stripe is self-contained and a reader that touches one stripe range reads only
that range's payloads.

A store emits **one or more** segments per stripe. The format places no limit on
segment count; how often a store cuts a new one is a writer policy, driven by a
target segment size and by the `uint32` ceiling on segment length. A store with
little data in a stripe produces a single segment; a store with large payloads
produces several.

```
  Nimble tablet
  +--------------------------------------------------------------+
  |  stripe 0                                                     |
  |      stream: id         Int64                                 |
  |      stream: payload    Int64     <- blob ids, not the bytes  |
  |      stream: tags       String                                |
  |      store 0, blob segment 0      <- payload bytes            |
  |      store 0, blob segment 1      <- same store, next segment |
  |      store 1, blob segment 0                                  |
  |      store 0, entry list          <- entries for this stripe  |
  |      store 1, entry list                                      |
  +--------------------------------------------------------------+
  |  stripe 1                                                     |
  |      stream data ...                                          |
  |      store 0, blob segment 2                                  |
  |      store 1, blob segment 1                                  |
  |      store 1, blob segment 2                                  |
  |      store 0, entry list                                      |
  |      store 1, entry list                                      |
  +--------------------------------------------------------------+
  |  stripe group 0 metadata                                      |
  |  ... more stripes and stripe groups ...                       |
  +--------------------------------------------------------------+
  |  blob.metadata              the manifest, an optional section |
  +--------------------------------------------------------------+
  |  stripes metadata                                             |
  |  footer                                                       |
  |  postscript                                                   |
  +--------------------------------------------------------------+
```

Segment ids are scoped to their store and number consecutively from zero across
the whole file, so a segment is identified by `(blobStoreId, segmentId)` in the
same way a payload is identified by `(blobStoreId, blobId)`.

A blob segment is a run of payload bytes with no internal framing: payloads are
concatenated back to back, and only the entry list records where each one
begins. A payload never straddles a segment boundary. The tablet's postscript,
footer, and chunk headers are unchanged.

A store's entry list for a stripe is written immediately after that stripe's
payload bytes, not gathered at the end of the file. Keeping the two together
means a reader that wants payloads from one stripe fetches the entries and the
bytes they describe from the same region, and never reads metadata for stripes
it is skipping.

Only `blob.metadata` is a named optional section. The payload regions and the
entry lists are anonymous byte ranges, reachable only by following offsets
recorded in the manifest.

## Metadata

The manifest is stored in the `blob.metadata` optional section. It is a list of
**blob groups**, one per blob store per stripe, and everything needed to read
that store in that stripe hangs off its group.

| Record | One per | Purpose |
|--------|---------|---------|
| `BlobGroup` | store and stripe | Owns the segments and the entry list for one store in one stripe |
| `BlobSegment` | payload run | Where a run of payload bytes lives in the file |
| `BlobEntry` | blob | Where one payload lives inside a segment |

Segments are held **inside** the group. There are only a handful per group, so
inlining them costs nothing, and they no longer need to name their store or
their own id -- the group supplies the store, and a segment's position in the
list is the id `BlobEntry::segmentId` refers to.

Entries are **not** inlined. Their count is proportional to the number of blobs,
so the group holds only the address of an entry list stored elsewhere in the
file. That is what keeps the manifest small enough to parse on every open.

```
   blob.metadata                            <- small, parsed on open
   +--------------------------------------+
   | version = 1                          |
   | groups[]                             |
   |   BlobGroup                          |
   |     blobStoreId = 0                  |
   |     stripeId    = 3                  |
   |     segments[]                       |
   |       { fileOffset 200000, 3 MB }    |---> payload bytes
   |       { fileOffset 400000, 2 MB }    |---> payload bytes
   |     entryCount = 5000                |
   |     entries = {offset 40960,         |
   |                size   180000}  ------|----+
   +--------------------------------------+    |   a file offset,
                                                |   not the entries
   at byte 40960, next to the payloads          |
   +--------------------------------------+ <--+
   | BlobEntries                          |
   |   [ BlobEntry, BlobEntry, ... 5000 ] |    <- read only when this
   +--------------------------------------+       store's blobs are wanted
```

`(blobStoreId, stripeId)` is what makes a group addressable. A reader decoding a
stripe selects the groups for that stripe, loads only their entry lists, and
never touches metadata belonging to stripes it is skipping. A store appears once
per stripe it has blobs in.

Grouping also scopes the lookups. Resolving a blob means searching the segments
of one group rather than every segment in the file, and once segment ids are
positions in that list the search becomes an array index.

`BlobEntries` is only a FlatBuffers envelope: a vector needs a root table to be
serialized, and that table is all it is. There is no `BlobEntries` type in C++,
where an entry list is a `std::vector<BlobEntry>` produced by
`serializeEntries` and `deserializeEntries`.

The `entries` field is a `MetadataSection` rather than an inline `BlobEntries`
table because FlatBuffers offsets are relative to the buffer being parsed and
cannot point at another region of the file. Nesting the table would put every
entry inside the manifest and make parsing it as expensive as reading all the
entries. `MetadataSection` is the same file-pointer type Nimble uses for
`Footer.stripes` and `Footer.stripe_groups[]`.

Entry lists are written through the tablet's metadata-section path, so they are
compressed automatically once they exceed the writer's compression threshold.
Payload bytes are not: they are written raw.

## Reading a payload

```
  (storeId, blobId)
        |
        +--> BlobEntry     -> segmentId, offsetInSegment, size, checksum
        |
        +--> BlobSegment   -> fileOffset
        |
        +--> read fileOffset + offsetInSegment, verify checksum
```

Two indirections, deliberately. The entry says where a payload sits inside its
segment; the segment says where that segment sits in the file. Separating them
means segments can move -- to a different stripe, or eventually out of the
tablet entirely -- without rewriting millions of entries.

A scan that does not project a blob-backed column never reads the blob region.

## Writing

The writer decides per value stream whether a column is blob-backed. For a
blob-backed column, every non-null payload is appended to the column's blob
store, which returns the id written into the value stream. Appended bytes
accumulate until the store cuts a segment.

A store cuts a segment when either of two things happens:

- the buffered bytes reach the writer's target segment size, or
- the stripe ends, which flushes whatever is buffered.

The first bounds writer memory and keeps segments at a size that reads and
compresses well; the second is what keeps a stripe self-contained. Each flush
records a `BlobSegment` covering the rows it holds, and subsequent payloads
begin a new segment.

At the end of a stripe, after the last segment is flushed, each store writes the
entry list for the blobs it added during that stripe and records a
`BlobGroup` for it. At close the writer emits only the manifest.

Ordering within a stripe is forced: payloads first, then the entry lists that
reference them, because each step records the offsets of the previous one. The
manifest comes last, since it records the offsets of everything.

## Current state and open questions

The format described here is what the metadata supports. The implementation is
not yet complete:

- Segments are flushed once at close rather than at stripe boundaries or on a
  size target, so each store emits exactly one segment for the whole file, the
  writer holds a whole column's payloads in memory, and `firstRow` and
  `rowCount` describe the whole file rather than a stripe. The writer already supports
  repeated flushes and assigns segment ids accordingly; only the policy that
  decides when to call it is missing.
- Blob payloads are always uncompressed. `BlobSegment` carries a compression
  type and `BlobEntry` carries compressed and uncompressed sizes, which describe
  two different compression granularities; which one applies is undecided.
- Entry lists are per store rather than per store and stripe, are written at the
  end of the file rather than next to the payloads they describe, and are loaded
  eagerly for every store when a reader opens a blob-backed file, regardless of
  projection.
- Entry lookup is a linear scan even though entries are dense and ordered by
  blob id.
- Only top-level `VARCHAR` and `VARBINARY` columns are supported. There is no
  deduplication of identical payloads.

The metadata is already shaped for per-stripe payloads: `BlobGroup` is keyed by
`(blobStoreId, stripeId)`, so a writer that flushes at stripe boundaries emits
one group per store per stripe and both the segments and the entry list follow.
What is missing is the writer doing it -- today a single group per store is
emitted at close with `stripeId` left at zero.
