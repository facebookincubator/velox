Nimble Java Reader & Writer
===========================

Using BlockBitPacking (EncodingType=15) as a concrete example

1 Write Path
------------

There is no pure Java Nimble writer . Spark writes Nimble files via JNI, delegating to the C++ Writer for encoding and file layout.

Java AlphaPageWriter.writePage(Page)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Serializes Presto Page → ByteBuffer


JNI JniAlphaWriter.writePageJni(writerId, buffer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xldb/alpha_jni/cpp/AlphaWriterJNI.cpp


C++ Writer::write(RowVector)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encoding selection → BlockBitPacking / Dictionary / RLE / ... → flush to disk

Implication:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Layer
     - Key Class
     - File
   * - Java
     - AlphaPageWriter
     - xldb/alpha_jni/alpha-common/.../writer/AlphaPageWriter.java
   * - Java
     - JniAlphaWriter
     - xldb/alpha_jni/alpha-common/.../writer/JniAlphaWriter.java
   * - JNI
     - AlphaWriterJNI.cpp
     - xldb/alpha_jni/cpp/AlphaWriterJNI.cpp
   * - C++
     - Writer
     - velox/dwio/nimble/writer/Writer.h

2 Read Path
-----------

Unlike the writer, the Java Nimble reader is a pure Java implementation in alpha.encodings . It is the production decoder used by Spark via xldb-orc → shaded AlphaRecordReader .

Three Java Nimble Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three Java packages related to Nimble encoding. Only one is production:

Production

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Package
     - Location
     - Consumers
   * - com.facebook.presto.alpha.encodings
     - presto-facebook-alpha/.../alpha/encodings/
     - Spark reads via AlphaRecordReader

Dead — prototype + its test JNI bridge

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Package
     - Location
     - Role
   * - com.facebook.presto.nimble
     - presto-facebook-alpha/.../nimble/
     - Selective-read prototype decoders — never shipped, zero consumers
   * - com.facebook.xldb.nimble.jni
     - xldb/alpha_jni/nimble-encodings/
     - JNI bridge that encodes test data via C++ for the prototype ’ s tests ( < scope > test < /scope > )

The prototype ’ s tests call NimbleEncodings.encode() via JNI to generate C++-encoded data, then verify the prototype ’ s Java decoders can read it. No alpha.* code imports xldb.nimble.jni .

Important:

alpha.encodings

There are also two separate EncodingType enums:

* alpha.common.EncodingType — production used by the read path; EncodingFactory calls fromValue(15) to dispatch decoding. Must be updated for new encodings.
* xldb.nimble.jni.EncodingType — test-only only imported by com.facebook.presto.nimble test files. Updated defensively to keep integer values in sync, but not required for production.

Java AlphaRecordReader
^^^^^^^^^^^^^^^^^^^^^^

Spark → xldb-orc → AlphaRecordReader


EncodingFactory.deserializeEncoding()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reads encoding prefix → dispatches on EncodingType


BlockBitPackingEncoding.createEncoding(dataType, input, memCtx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dispatches to Byte / Short / Int / Long variant based on DataType


Constructor: parse nested metadata + decompress packed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reads 3 sub-streams via recursive EncodingFactory calls


materialize(rowCount) → Block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-block bit-unpack + baseline addition → ByteArrayBlock / IntArrayBlock / ...

BlockBitPacking Binary Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What the Java decoder reads

.. code-block:: text

    [Encoding Prefix]  encodingType=15, dataType, rowCount

    [compressionType]  uint8  — Uncompressed / Zstd / Zstrong
    [blockSize]         uint16 — default 1024
    [numBlocks]         uint16 — ceil(rowCount / blockSize)

    [size₁] [baselines encoding]   physicalType[numBlocks]
    [size₂] [bitWidths encoding]   uint8[numBlocks], 255 = skip-encoding
    [size₃] [offsets encoding]     uint32[numBlocks], byte offset per block

    [packed data]  all blocks contiguously, compressed with compressionType

Type Specialization
^^^^^^^^^^^^^^^^^^^

ByteBlockBitPackingEncoding

Int8 / Uint8 — baseline: byte , reads short for bit extraction

ShortBlockBitPackingEncoding

Int16 / Uint16 — baseline: short , reads int for bit extraction

IntBlockBitPackingEncoding

Int32 / Uint32 / Float — baseline: int , reads long for extraction

LongBlockBitPackingEncoding

Int64 / Uint64 / Double — baseline: long , two-word read for bw > 58

3 E2E Verification
------------------

The E2E test verifies that the pure Java decoder correctly reads what the C++ encoder writes. Artifacts are generated once by the C++ test generator and committed to the repo.

Step 1 — Generate Artifacts C++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test_generator ( //velox/dwio/nimble/encodings/tests:test_generator ) produces artifact files for each (encoding, dataType, rowCount) combination:

For each encoding × type × rowCount:

1. Generate random data — seeded RNG creates rowCount random values of the given type
2. Write .data file — raw bytes of each value in native byte order (no header, no encoding — just rowCount × sizeof(T) bytes)
3. Encode 3 × — calls the real C++ encoder ( E::encode() ) via a test wrapper ( nimble::test::Encoder < E > ::encode() ) with each compression type
4. Write .encoding files — the full encoded blob (prefix + nested metadata + compressed packed data) for Uncompressed, Zstd, and Zstrong

Real encoder, controlled selection:

BlockBitPackingEncoding::encode()

TestTrivialEncodingSelectionPolicy

.. code-block:: text

    # Generate only BlockBitPacking artifacts (use --encoding_filter)
    buck2 run //velox/dwio/nimble/encodings/tests:test_generator \
        -- --output_dir=presto-facebook-alpha/src/test/resources/encodings \
           --encoding_filter=BlockBitPacking

    # Output per (dataType, rowCount):
    #   BlockBitPacking_Int32_256.data                  ← raw values (ground truth)
    #   BlockBitPacking_Int32_256_Uncompressed.encoding ← encoded blob, no compression
    #   BlockBitPacking_Int32_256_Zstd.encoding          ← encoded blob, Zstd compressed
    #   BlockBitPacking_Int32_256_Zstrong.encoding       ← encoded blob, Zstrong compressed

    # Omit --encoding_filter to regenerate ALL encoding artifacts

Step 2 — Verify Java
^^^^^^^^^^^^^^^^^^^^

List data files in resources/encodings/
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parse filename → encoding type, data type, row count


Read .data file → expected values[]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Raw source values written by C++ (ground truth)


Read .encoding file → EncodingFactory.deserialize()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recursive Nimble decoding → encoding.materialize(rowCount) → decoded values[]


assertEquals(expected[i], decoded[i]) ∀ i
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3 compressions × 10 data types × 2 sizes = 60 test cases per encoding

Running the Test
^^^^^^^^^^^^^^^^

.. code-block:: text

    # Generate artifacts (only needed when C++ format changes)
    buck2 run //velox/dwio/nimble/encodings/tests:test_generator \
        -- --output_dir=presto-facebook-alpha/src/test/resources/encodings

    # Run the Java E2E test
    cd github/presto-facebook-trunk
    mvn test -pl presto-facebook-alpha \
        -Dtest=TestEncodingsE2E \
        -Dmaven.gitcommitid.skip=true \
        -Dcheckstyle.skip=true

Coverage:

60 test cases
