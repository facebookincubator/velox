# Adding a Required Feature to Nimble Files

A required feature is a format change that a reader must understand to read a
file correctly. Each one has a bit in the `required_features` field of the
file's `columnar.properties` section. A reader with this check refuses to open
a file that sets a bit it does not support, instead of returning wrong data.

This guide covers when a change needs a bit, how to add one, and how to roll it
out. The rules also appear next to the code: `RequiredFeature` and
`FileProperties::kSupportedRequiredFeatures` in `tablet/FileProperties.h`, and
the `required_features` field in `tablet/FileProperties.fbs`. Paths are
relative to `velox/dwio/nimble/`.

## When a Change Needs a Bit

A change needs a bit when a reader that does not know it would return wrong
data without noticing. Examples:

- The byte layout of an existing encoding or stream changes.
- Data moves somewhere older readers do not look.
- An existing field takes on a new meaning.
- An existing enum field gets a new value that older readers do not reject.

Prefer a bit even when older readers already fail with an error. A new
`EncodingType` value is such a change: Nimble's decoders throw only when they
reach a stream that uses it, possibly after the reader has returned rows, and
`EncodingFactory`'s error suggests garbage input. With a bit, readers that have
the check refuse the file when they open it, with `NOT_SUPPORTED`.

A change that only adds information, such as a new optional field at the end of
its table or a new optional section, needs no bit if older readers can safely
ignore it: readers skip FlatBuffers fields and sections they do not know. Newer
readers must treat its absence as unknown, because files written before the
change never carry it. A tool that copies it unchanged while rewriting the data
it describes leaves it stale. If newer readers rely on it to read correctly,
treat it as a required feature.

A bit can gate only what a reader parses after it checks the bits. Before that
check, `TabletReader`:

- parses the postscript and the footer, including the optional-section index
  and how each section is compressed;
- parses the stripes section;
- in a file with a single stripe group, parses that stripe group if the
  reader's speculative read of the file's tail already covers it;
- reads and decompresses the other optional sections it preloads.

A change to any of these that older readers would misread needs a check that
readers make earlier, such as a new major version in the postscript, which
`Postscript::parse` refuses when it is newer than its own.

The bit marks the whole file. If only some of the file's data uses the feature,
that data must also say so, and readers decode each piece by its own marker.
The bit is what stops older readers.

Properties added before `required_features`, such as
`compactRowCountEncoding`, have their own fields and no bit. A reader that
predates one of them can misread a file that uses it.

## How the Check Works

- **Writing.** `Writer::writeProperties` runs once per file, when the file is
  closed, and writes `columnar.properties` unless every property has its
  default value. `FileProperties::serialize()` writes a bit for each feature
  passed to the `FileProperties` constructor and leaves the field out when
  there are none. A file without the field, or without a `columnar.properties`
  section, requires no features; that includes every file written before
  `required_features` existed.
- **Reading.** `FileProperties::deserialize()` checks the bits before it reads
  any other property. If a set bit is not in
  `FileProperties::kSupportedRequiredFeatures`, it throws a `NimbleUserError`
  with error code `error_code::NotSupported` (`NOT_SUPPORTED`) and a message
  like:

  ```
  File requires features this reader does not support. Count: 1, lowest bit: 3
  ```

  `TabletReader` calls `deserialize()` when it opens a file, so a refused file
  fails before any stream is decoded. Code that parses `columnar.properties`
  itself must also go through `deserialize()`. What a refusal means to users
  depends on the caller: a caller that turns read errors into empty results
  turns a refusal into missing data.
- **Readers without the check.** A reader built before `required_features`
  existed never looks at the field. It reads the file as if no feature were
  set.
- **Data served by `NimbleIndexProjector`.** The projector serves a file's data
  in `kTablet` slices. For a whole stripe it passes the file's encoded streams
  through. For part of a stripe it slices them with `StreamSlicer`, which reads
  the encoded bytes and can re-encode them. Clients decode the slices with
  `Deserializer` (`serializer/Deserializer.h`) and never see `FileProperties`,
  so the check does not protect them. Step 2 covers this path.
- **Encoded data outside files.** The encoders that write files also write
  `Serializer` output, the slices the projector re-encodes, and external
  dictionary alphabets. None of these carries `required_features`. Step 4
  covers this.

## Step-by-Step Guide

### Step 1: Add the `RequiredFeature` Value

Give the new feature the next value after the last one, and write the value
out:

```cpp
enum class RequiredFeature : uint32_t {
  kMyFeature = 0,
};
```

Files on disk depend on these values. Never renumber, reuse or remove one.
Assign bits only in upstream Velox (facebookincubator/velox): a fork that
assigns its own bits will collide with later upstream features.

### Step 2: Handle the Feature in Every Read Path

A reader learns whether a file uses the feature from
`FileProperties::requiredFeatures()`. With `tablet` pointing to the file's
`TabletReader`:

```cpp
const auto& features = tablet->properties().requiredFeatures();
const bool usesMyFeature =
    std::ranges::find(features, RequiredFeature::kMyFeature) != features.end();
```

Handle the feature wherever the affected data is read. Find those places from
the code that reads the data:

- every place that builds an `Encoding` or an `EncodingView` from stream bytes:
  `EncodingFactory` and its subclasses, and `createEncodingView`;
- every place that reads encoded bytes without building one:
  `EncodingFactory::slice` (`EncodingSliceFactory` and each encoding's static
  `slice`), `StreamSlicer`, and code that reads `EncodingPrefix` fields
  directly.

The callers of an existing property are not a complete list, because some
decoders build encodings with default options and never read file properties.
For comparison, these read `FileProperties::compactRowCountEncoding()`:

- `BatchReader` (`reader/BatchReader.cpp`), for shared-dictionary streams only;
- `SelectiveNimbleRowReader` (`selective/SelectiveNimbleReader.cpp`);
- `SelectiveNimbleIndexReader` (`selective/SelectiveNimbleIndexReader.cpp`);
- `NimbleIndexProjector` (`index/NimbleIndexProjector.cpp`);
- `IndexScanCursor` (`index/IndexScanCursor.cpp`).

If the feature changes anything that projector clients decode or rely on, such
as stream bytes, chunk framing, what values or the schema mean, or where data
lives, `NimbleIndexProjector` needs one of two treatments:

- Refuse such files when the projector is created, with `NIMBLE_UNSUPPORTED`,
  as it does for shared dictionaries. Callers then see every read of such a
  file fail, so check what they do with failed reads. Clients cannot see the
  bit, so removing the refusal later is itself a format change for them; keep
  it until the second treatment replaces it.
- Or carry the feature in the `kTablet` format in a way older clients reject,
  and give `Deserializer` and the projector's slicing the read path. Older
  clients ignore slice-header flag bits they do not know, so a new flag alone
  is not enough. A new `SerializationVersion` works only with a byte value that
  no client has ever accepted. Version bytes have been retired and reassigned
  before, so pick a value above every value ever used. Emit it for every slice
  of a file that sets the bit, and only for those.

### Step 3: Declare Reader Support

Append the feature to `FileProperties::kSupportedRequiredFeatures`, and append
it by enumerator name to `kAssignedRequiredFeatures` in
`tablet/tests/FilePropertiesTest.cpp`, growing each array by one.
`requiredFeaturesAreAppendOnly` fails unless the two lists match and each value
equals its position. It cannot see an enumerator missing from both lists, so
land Steps 1 to 3 together.

`kSupportedRequiredFeatures` speaks for every program built from this library.
Adding the feature stops `TabletReader` from refusing files with the bit in all
of them, including tools. Before you add it:

- Step 2 must cover every reader.
- Code that decodes stream bytes on its own path must handle the feature, or
  compare `requiredFeatures()` with its own list of handled features and refuse
  files with any other bit.
- A tool that copies stream bytes into another file must write the bits of the
  features those bytes use into the output's `columnar.properties`, along with
  every other property they depend on. Otherwise it must refuse inputs whose
  `requiredFeatures()` lists a feature it does not handle. A tool that changes
  what the input's `columnar.properties` describes must not copy that section
  unchanged.

Copy tools and decoders outside the main read paths are safest when they refuse
every feature they do not list, from the start, so a new feature makes them
fail rather than misread.

### Step 4: Set the Bit in the Writer

Put the writer change behind a `WriterOptions` flag that is off by default, as
`experimentalCompactRowCountEncoding` is. Keep the flag out of `nimble::Config`
until rollout step 2 is done: `nimble::Config` entries can be set from a
table's serde parameters. Then:

- Record whether the file actually uses the feature where its bytes are
  produced, rather than deriving it from the options alone. A file that never
  used the feature then stays readable by older readers. Streams are encoded on
  `encodingExecutor` threads when encoding is parallel, so make the record
  thread-safe there.
- Make the flag gate every way the feature can be produced, including encoding
  layouts the writer replays (`encodingLayoutTree`, or a layout cached from a
  stream's first encode) and options that pin an encoding to specific columns.
- In `Writer::writeProperties`, add the feature to the condition of the early
  return. Without it, a file whose only property is this feature gets no
  `columnar.properties` section, and its bit is lost.
- Pass the features to the `FileProperties` constructor in increasing bit
  order, each once. The constructor checks this.

Keep a change to an encoding's bytes behind an `Encoding::Options` field that
only the file writer sets, and only when it records the bit. Any other producer
of encoded data, such as `Serializer`, needs its own versioned format and
rollout before it uses the change.

`TabletWriter` writes only the properties its caller gives it, and some callers
give none. If the feature comes from `TabletWriter`, make `TabletWriter` refuse
to produce it unless its caller records the bit. `Writer` writes properties in
the close callback, before the chunk statistics root, the stripes section and
the footer, so whether the file uses the feature must be known by then.

`Writer::suspend()` and `Writer::resume()` are not implemented yet. Whoever
implements them must make a suspended file carry the bit for the stripes it
holds, and make `resume()` restore whether earlier stripes used the feature.

### Step 5: Test

- A file written with the feature sets its bit and reads back correctly through
  every read path from Step 2, including partial-stripe projection if the
  projector serves such files.
- A file whose only property is the feature has a `columnar.properties` section
  with the bit.
- With the writer flag on, a file that never uses the feature has no bit.
- With the writer flag off, the writer does not produce the feature, even when
  it replays a layout captured from a file that uses it.
- A file written without the feature has no `required_features` field.

Writer tests for properties live in `tests/WriterTest.cpp`. The tests that need
an unsupported bit use bits from 64 up: the `fromBit64()` cases and the bit-100
case in `tablet/tests/FilePropertiesTest.cpp`, and
`TabletTest.rejectsUnsupportedRequiredFeatures` in
`tablet/tests/TabletTest.cpp`. If assigned bits ever reach 64, move those cases
higher.

## Rollout

Readers go first. A reader that lacks the feature refuses a file with its bit,
and a reader built before `required_features` existed reads that file wrong.

1. **Land Steps 1 to 5 with the writer flag off.** Files are written exactly as
   before.
2. **Wait for readers.** Every program that reads the data must run a build
   with the feature's read path before any writer sets the bit. That includes:
   - tools, including those that copy streams;
   - readers in other languages or implementations;
   - clients that decode data from `NimbleIndexProjector`, unless the projector
     refuses files with the bit and every caller can live with those reads
     failing (see Step 2).
3. **Turn the writer flag on gradually**, writer by writer through each
   writer's own options. Changing the flag's default turns it on at once in
   every writer that uses the defaults. Watch for the message
   `File requires features this reader does not support`, including in callers
   that turn read errors into empty results: each one is a reader that step 2
   missed. A quiet ramp does not prove readers are ready, because readers built
   before `required_features` never refuse.
4. **Keep reader support forever.** Files with the bit stay on disk. To roll
   back, turn the writer flag off; the files already written still need their
   readers. Never remove the read path or reuse the bit, and never roll a
   reader back past the build that added the feature: a build without it
   refuses those files, and a build from before `required_features` reads them
   wrong.

Readers in other implementations should add the same check, so that later
features make them fail instead of misread.

What a reader does with each kind of file:

| Reader | File without the bit | File with the bit |
|---|---|---|
| Built before `required_features` existed | reads it | ignores the bit, so it can return wrong data |
| Has the check, lacks the feature | reads it | refuses to open it (`NOT_SUPPORTED`) |
| Supports the feature | reads it | reads it |
