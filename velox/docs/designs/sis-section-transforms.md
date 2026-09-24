# Reversible Section Transforms for SubIntSplit

## Motivation

SubIntSplit cuts an integer column into contiguous bit-range sections and gives each its own
child encoding. A section that compresses poorly when its rows are read in the column's row
order can compress well once rows are reordered by the value of another section: sorting a
column-like sub-part by a related key groups similar values together, which is exactly what
slowly-varying-friendly child encodings such as RLE and MainlyConstant need.

Row order is therefore a lever SubIntSplit can pull without changing what each section stores,
as long as the reorder is reversible and the original row order can still be recovered on read.
This is why each SubIntSplit section can carry its own transform: a section is reordered only
when doing so is cheaper than leaving it alone.

The reorder has to be undoable independently of which sections chose it. Some sections in a
block may be transformed and others left as-is, so the mechanism that recovers original row
order cannot depend on every section agreeing to participate; it has to be reconstructable from
a single, always-present source of truth. That is what motivates storing one section unpermuted
as the sort key described below, rather than, say, storing an explicit permutation array
alongside the block.

## The two current transforms

Every other transform id that has existed in this design is retired; see below.

### Key-derived permutation (transform id 1)

One section in the block is designated the **key** section. Every other section that selects
this transform is stably sorted by the key section's decoded value: rows with equal keys keep
their relative order, which is what lets a reader reconstruct the same permutation deterministically
from the key values alone.

The key section itself is always stored **unpermuted**, since it is what a reader uses to
rebuild the mapping back to original row order; a permuted key section would have nothing to
rebuild it from.

At encode time this means: pick a key section (typically the one whose value correlates with
where the gains lie in the other sections), compute a stable sort of row indices by that
section's values, then gather every other transformed section's values through that
permutation before handing them to child-encoding selection. The key section's own bytes are
never touched by the permutation.

At decode time, the process runs in reverse: the key section is decoded first, in its stored
(unpermuted) order, and the same stable sort is recomputed over it to get back the permutation
that was used at encode time. Any other section flagged as transformed is then decoded in its
stored (sorted) order and gathered back through the inverse of that permutation to restore
original row order.

### Row frame (header flag bit 1)

A row frame fits a line, `slope * row + base`, to the whole column and subtracts it from every
value before the column is split into sections. Where the fit is a poor match for value drift,
a step frame (a small number of flat segments rather than one line) is used instead. The frame
is a property of the whole column, not of one section, and is recorded once in the header rather
than per section.

## Wire format

Encoding type 26 (`SubIntSplitReordered`) extends the SubIntSplit prefix with:

```
[standard SubIntSplit prefix]
[1B]  numSections
[1B]  flags                 bit0 = delta
                             bit1 = row frame present
                             bit2 = transforms present
[17B] row frame              present only if flags.bit1
      [1B]  guard
      [8B]  slope
      [8B]  base
[transform block]            present only if flags.bit2
      [1B]  keySection       index of the section used as the sort key
      [numSections x 1B]  transformId   per section, 0 = none
      [per transformed section]
          [4B]  codebookSize
          [codebookSize x 8B]  codebook entries
[numSections x 6B]  {bitStart, bitEnd, encodedSize}
[section payloads...]
```

Field-by-field:

* `numSections` and the trailing `{bitStart, bitEnd, encodedSize}` array are the ordinary
  SubIntSplit section table; nothing about them changes when transforms are in use.
* `flags` is checked before anything else in the header is parsed, since it determines whether
  the row-frame and transform blocks are present at all. A stream with `flags == 0` (delta off,
  no row frame, no transforms) parses identically to a plain SubIntSplit stream past the prefix.
* The row frame, when present, is fixed-size (17 bytes) regardless of column width, since
  `slope` and `base` are stored as 8-byte values and adjusted for the column's actual type on
  read; the guard byte lets a reader sanity-check the frame before applying it.
* The transform block's `transformId` array is one byte per section, including the key section
  itself (which always reads back as `0`, since it is never transformed) and any section that
  chose no transform (also `0`).
* The codebook that follows a transformed section exists so a future value-remapping transform
  can attach an alphabet without a further wire change; `codebookSize` of `0` for the key-derived
  permutation is what makes it a no-op for that transform today.

Two things this format deliberately does not have:

* No separate "transform block size" field. A reader that knows `numSections` and each
  section's `transformId` and codebook size can compute the block's length by walking it, so a
  redundant size field was dropped along with the transforms that needed it (see below).
* No per-transform block state carried alongside the header. Retired transforms needed extra
  bookkeeping of this kind; the two current transforms do not, so none is reserved for it.

## How reads address rows

A read that needs row *r*'s value in a permuted section cannot index directly into that
section's storage, because the section holds rows in sorted-by-key order, not original order.
The reader instead rebuilds a **position map** from the key section's decoded values (the same
stable sort used at encode time), and resolves the probe through it: one indirection from
original row number to the row's position within the section.

Because building the position map requires decoding the key section, the whole permuted column
is decoded once per block and the decode is cached in the view rather than recomputed for every
probe. A single point access pays for the decode and the map; repeated point or range access
into the same block reuses both.

This gives the transform two different cost profiles depending on the access pattern:

* **Bulk / sequential decode** never needs the position map at all. Rows are assembled in their
  stored (sorted-by-key) order exactly like an untransformed block, and the whole assembled
  block is gathered through the inverse permutation once at the end. The per-row cost of the
  transform is a single gather, independent of how many sections were transformed.
* **Point / scattered access** pays for decoding the key section and building the position map
  the first time a block is touched, then one indirection through that map per probe
  afterwards. This is why the position map is cached rather than treated as a temporary: without
  the cache, every probe into a fresh row of an already-visited block would redo the same
  key-section decode and sort.

A reader that cannot cache across probes (for example, because it is only ever asked for one
row from a block) still gets a correct answer, just without the amortization; correctness never
depends on the cache being present.

## How selection decides

For each section, split selection prices the section twice: once with the section's data left
in original row order, and once with it passed through the section's transform. Whichever
priced encoding is cheaper is what gets selected; a section that does not benefit from being
reordered is simply not transformed, regardless of whether other sections in the same block are.

Auto-selection only prices the key-derived permutation transform against the untransformed
alternative. The row frame is not decided this way: it is applied when a slope/step fit exists
and is judged beneficial for the column as a whole, ahead of section splitting, rather than
priced per section. The retired transforms below are never priced, since selection no longer
knows about them.

## Retired transforms

Transform ids 2 to 4 (value relabelling variants), 5 to 6 (Burrows-Wheeler-style transforms),
and 7 (bit-plane transform) have been retired. Current readers reject these transform ids
outright rather than attempting to decode them. They were dropped because selection never
priced them out as cheaper than the alternatives it already had, and because they imposed a
poor point-access cost: transforms in the Burrows-Wheeler family, in particular, require more
than one indirection (extra decode work per probe) to resolve a single row, unlike the
key-derived permutation's single indirection through its position map.
