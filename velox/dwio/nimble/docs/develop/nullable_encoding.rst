NullableEncoding — Nimble
=========================

Nimble encoding metadata
^^^^^^^^^^^^^^^^^^^^^^^^

Nimble file structure — how the reader finds encoding data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Physical layout on disk (read from end):

Stripe 0 data

raw stream bytes

Stripe 1 data

raw stream bytes

...

StripeGroup

per-stream offsets

Stripes meta

stripe boundaries

Optional

schema, stats

Footer

272 B

Postscript

20 B

Stage 1 — Metadata (small reads from end of file)

Postscript

Footer

StripeGroup

Optional sections

Gives you file_info, schema, stripes, streams tables — all without touching stripe data.

Stage 2 — Stream data (reads actual stripe bytes)

encodings table

traverseEncodings

Key insight:

self-describing

Binary format — how each encoding is parsed from raw bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every encoding starts with a 6-byte header:

byte 0

EncodingType

byte 1

DataType

bytes 2–5

row count (uint32)

bytes 6+

encoding-specific payload

EncodingType byte values:

0x00 Trivial

0x07 SparseBool

0x0A Nullable

0x05 MainlyConstant

0x03 Dictionary

0x06 Constant

0x02 FixedBitWidth

0x0C CBP

Payload format is hardcoded per encoding type:

Nullable

4B data size

Data child bytes

Nulls child bytes

2 children. Nulls size = total - 4 - data size.

MainlyConstant

4B IsCommon size

IsCommon bytes

4B OtherVal size

OtherValues bytes

2 children. CommonValue is inline in the header.

SparseBool

1B sparseValue

remaining = Indices child bytes

1 child. sparseValue: true=indices point to set bits, false=unset bits.

Trivial < String >

1B compress

4B lengths size

Lengths child bytes

blob (string data)

1 child (Lengths). Blob is raw/compressed string bytes.

CBP / FBW / Const

payload only — no children (leaf)

0 children. Entire blob after header is the encoded data.

The parser ( traverseEncodings ) and the reader ( NullableEncoding constructor) use the exact same byte layout knowledge.

is

EncodingFactory::create()

NullableEncoding
^^^^^^^^^^^^^^^^

Example: 50 rows, only row 37 is non-null
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NULL

non-null value

Logical column (50 rows):


NullableEncoding splits this into nulls\_ + nonNullValues\_

nulls\_ — one bool per row (T=non-null, F=null):

nonNullValues\_ — only 1 entry:

42

The value at row 37. No gaps, no placeholders.

How Nulls Are Stored
^^^^^^^^^^^^^^^^^^^^

Option A: TrivialEncoding < bool > 7 bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stores 1 bit per row. 50 rows = 50 bits = 7 bytes.

Packed bitmap (bit 37 = 1, all others = 0):

49 zero-bits stored just to record "not null". All wasted.

Option B: SparseBoolEncoding ~5 bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stores only positions of the minority value. 1 non-null → store [37].

On disk:

sparseValue\_ = true

+

indices [37]

1 byte (sparseValue) + ~4 bytes (one uint32 index). Done.

sparseValue\_

sparseValue\_ = true

sparseValue\_ = false

Storage comparison as data grows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All examples: 1 non-null row. Only the total row count changes.

50 rows

TrivialEncoding: 7 bytes (50 bits)

SparseBool: ~5 bytes (1 index)

1.4x smaller

1,000 rows

Trivial: 125 bytes (1,000 bits)

~5 B

25x smaller

1,000,000 rows

Trivial: 122 KB (1M bits)

~5 B — 25,000x smaller

2.9B rows (your case)

Trivial: 345 MB (2.9B bits)

345 MB

258K non-nulls

SparseBool

~750 KB — 460x smaller

When does SparseBool win?

1/lg(n)

3.2%

0.009%

When does Trivial win?
