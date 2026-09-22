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

The schema says so directly: a blob-backed column is its own node kind, written
as `BlobString` or `BlobBinary` depending on the type it presents as. The kind
is what tells a reader to build a blob-aware field reader, and it is also what
stops an older reader from misreading the file -- an unknown kind fails at
schema parse rather than decoding blob ids as text.

Because the kind already says the stream holds ids, the node declares its
stream as `Int64` rather than claiming to be a string stream. A
`nimble.blob_store.id` attribute on the same node records *which* store backs
the column, which is the one thing the kind cannot carry.

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

A segment id is a segment's position in its group, so it restarts at zero in
every stripe. Blob ids do not: they run for the life of the store, so a stale
stripe id surfaces as a missing blob rather than the wrong payload.

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

A group is identified by its `(blobStoreId, stripeId)` pair, and a store appears
once per stripe it has blobs in. Groups are stored ordered by `stripeId` then
`blobStoreId` -- the order a writer produces naturally, since stripes are
written in sequence and each stripe emits its stores. Keeping that order makes
every group of a stripe adjacent, so a reader decoding a stripe can binary
search for its groups, load only their entry lists, and never touch metadata
belonging to stripes it skips. The parser enforces the order, which also rules
out duplicate groups.

Grouping scopes the lookups too. Resolving a blob searches the segments of one
group rather than every segment in the file, and because a segment's id is its
position in that list, the search is an array index.

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
Payload bytes have their own scheme, below.

## Compression

There are two compression layers, configured per store and independent of each
other. A store may use either, both, or neither.

```
  payloads      #### #### ####   #### #### ####   #### #### ####
                     |                |                |
  per blob      (skipped)         [f] [f] [f]      [f] [f] [f]
                     |                |                |
  per segment   (skipped)         (skipped)        [ one frame ]
                     |                |                |
  on disk       #### #### ####    [f] [f] [f]       [~~~~~~~~~~]
```

The inner layer reduces each payload on its own. It gives up cross-payload
redundancy to keep random access: expanding one payload does not touch the rest
of the segment, which matters when a scan wants a handful of rows out of many.

The outer layer reduces the segment as a whole, over whatever the inner one
produced. It gives the codec the most to work with and is right when a scan
reads most of a stripe.

Using both is a real combination rather than a contradiction. The inner layer
keeps each blob independently addressable in the segment's restored image, and
the outer one shrinks what has to be stored and fetched to produce that image.
It costs a second pass over bytes the first pass already reduced, which is
worth it when the payloads share structure the per-blob frames could not see.

What the metadata records is where compression actually landed:
`BlobSegment.compressionType` for the outer layer, `BlobEntry.compressionType`
for the inner one. Either may come back uncompressed, because the codec
declines whenever it would not shrink the bytes -- and at the inner layer that
decision is made payload by payload, so one segment can hold both kinds.

Offsets always address the segment's uncompressed bytes, meaning the inner
layer rather than the original payloads. A blob's checksum covers the payload
the reader ends up with, so it is verified after both expansions.

A reader loads a group's entry list the first time it resolves a blob in that
group, not when it opens the file. Opening therefore costs one manifest read
however many stripes the file has, and a scan pays only for the stripes and
columns it touches.

## Reading a payload

```
  (storeId, stripeId)                      the reader knows both: the stripe it
        |                                  is decoding, and the store from the
        |                                  column's blob node
        +--> BlobGroup     -> segments[], entry list location
        |
  blobId
        |
        +--> BlobEntry     -> segmentId, offsetInSegment, size, checksum
        |
        +--> group.segments[segmentId]  -> fileOffset
        |
        +--> read fileOffset + offsetInSegment, expand, verify checksum
```

Two indirections below the group, deliberately. The entry says where a payload
sits inside its segment; the segment says where that segment sits in the file.
Separating them means segments can move -- to a different stripe, or eventually
out of the tablet entirely -- without rewriting millions of entries.

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
compresses the buffered payloads through the store's layers, records a
`BlobSegment` covering the rows it holds, and starts a new segment.

Compressing at flush rather than at append is what lets the buffered bytes stay
raw. The writer reads them back to collect column statistics, and a store that
compressed on the way in would have nothing but frames to offer.

At the end of a stripe, after the last segment is flushed, each store writes the
entry list for the blobs it added during that stripe and records a
`BlobGroup` for it. At close the writer emits only the manifest.

Ordering within a stripe is forced: payloads first, then the entry lists that
reference them, because each step records the offsets of the previous one. The
manifest comes last, since it records the offsets of everything.

## Current state

The implementation is not yet complete. None of what is left needs a format
change:

- A store can cut several segments within a stripe, but nothing does so on its
  own: a segment is flushed only when the stripe ends. Payload bytes do count
  towards the writer's memory, so a blob-heavy column shortens the stripe rather
  than growing without bound, but there is no target segment size.
- Only top-level `VARCHAR` and `VARBINARY` columns are supported. There is no
  deduplication of identical payloads.
- `BlobSegment.firstRow` and `rowCount` are recorded but nothing reads them.
  They are there for a reader that wants to fetch a row range's payloads
  without walking the entry list first.
