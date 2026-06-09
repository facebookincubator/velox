# Parquet reader test fixtures

This directory contains small Parquet files and serialized page-header samples
used by Parquet reader tests. The files are intentionally diverse: some are
normal scalar data sets, some exercise legacy or malformed nested encodings, and
some preserve edge cases found in external writers.

Unless noted otherwise, the metadata below was checked with:

```bash
parquet-tools inspect velox/dwio/parquet/tests/examples/<file>
```

The metadata summary lists row count, row group count, and the leaf columns as
reported by `parquet-tools inspect`, including physical type, logical or legacy
annotation, definition level, repetition level, and compression when useful.

## Scalar and filtering fixtures

### `sample.parquet`

- Metadata: `created_by=parquet-cpp version 1.5.1-SNAPSHOT`, 20 rows, 2 row
  groups, columns `a: INT64` and `b: DOUBLE`, both optional, uncompressed.
- Purpose: General smoke-test fixture for scan, lazy loading, count(*), range
  reads, file preloading, row-group offsets, and integer/double filters. Data is
  `a = 1..20` and `b = 1.0..20.0`; each row group has 10 rows.

### `sample_with_rowindex.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 15.0.1`, 20 rows, 1 row group,
  columns `a: INT64`, `b: INT64`, and `_tmp_metadata_row_index: INT64`, Snappy.
- Purpose: Verifies behavior when a file already contains a column with the same
  name as Velox's generated row-index metadata column. Tests must use the user
  data column instead of synthesizing one.

### `gcc_data_diff.parquet`

- Metadata: `created_by=parquet-mr version 1.13.1`, 126 rows, 1 row group,
  columns `c1: INT64`, `c2: INT64`, `c3: INT64`, Snappy. `c3` is required while
  `c1` and `c2` are optional.
- Purpose: Supports aggregate-pushdown coverage for filtered scans and grouped
  aggregation over Parquet data.

### `different_encodings_with_filter.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 15.0.0`, 15 rows, 3 row
  groups, columns `n_0: INT32`, `n_1: INT32`, `n_2: BYTE_ARRAY String`, Snappy.
- Purpose: Exercises filter evaluation when row groups and columns use different
  encodings, including nullable string results.

### `nation.parquet`

- Metadata: `created_by=parquet-mr version 1.11.1`, 25 rows, 1 row group,
  columns `nationkey: INT64`, `name: BYTE_ARRAY String`, `regionkey: INT64`,
  `comment: BYTE_ARRAY String`, Gzip.
- Purpose: TPC-H `nation` fixture used for string filters, binary-as-varchar
  reads, and metadata-column type mismatch regression tests.

### `str_sort.parquet`

- Metadata: `created_by=parquet-cpp-velox`, 1000 rows, 1 row group, 11 columns,
  including `query_sig: BYTE_ARRAY String`, `result_sig: BYTE_ARRAY String`,
  integer counters, and double expected counters, all uncompressed.
- Purpose: Benchmark fixture for `RowContainerSortBenchmark`; only
  `query_sig` and `result_sig` are read by the benchmark setup.

### `multiple_row_groups.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 499 rows, 4 row groups,
  column `id: INT64`, required, Gzip.
- Purpose: Tests row-group prefetching, buffering, eviction, and multi-row-group
  scan behavior.

### `empty_row_groups.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 11.0.0-SNAPSHOT`, 5 rows, 4
  row groups, column `a: INT32`, uncompressed.
- Purpose: Regression fixture for files that contain empty row groups. Expected
  values in tests are `{0, 3, 3, 3, 3}`.

### `empty.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 0 rows, 0 row
  groups, 2 columns, format version 1.0.
- Purpose: Verifies that Velox can construct schema information and report zero
  rows for an otherwise valid empty Parquet file. The logical schema is
  `a: BIGINT, b: DOUBLE`.

## Numeric, decimal, date, and timestamp fixtures

### `int.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 5.0.0`, 10 rows, 1 row group,
  columns `int: INT32` and `bigint: INT64`, Snappy.
- Purpose: Basic integer fixture. Data is `int = 100..109` and
  `bigint = 1000..1009`; used for multiple-filter tests.

### `uint.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 13.0.0`, 3 rows, 1 row group,
  columns `uint8: INT32 UINT_8`, `uint16: INT32 UINT_16`,
  `uint32: INT32 UINT_32`, `uint64: INT64 UINT_64`, Snappy.
- Purpose: Tests unsigned Parquet logical integer annotations and widening into
  Velox signed integer and decimal output types.

### `date.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 5.0.0`, 25 rows, 1 row group,
  column `date: INT32 Date`, Snappy.
- Purpose: Date logical type fixture. Data spans Velox date offsets `-5..19`,
  corresponding to `1969-12-27..1970-01-20`, and is used for date filters.

### `decimal.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 9.0.0`, 20 rows, 2 row groups,
  columns `a: FIXED_LEN_BYTE_ARRAY Decimal(5,2)` and
  `b: FIXED_LEN_BYTE_ARRAY Decimal(20,5)`, Snappy.
- Purpose: Tests short and long decimal reads, decimal filter pushdown, and
  reading file decimals with larger requested precision. Data is
  `a = 100.01..100.20` and `b = 100000000000000.00001..00020`.

### `decimal_dict.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 6 rows, 1 row group,
  columns `a: INT32 Decimal(7,2)` and `b: INT64 Decimal(14,2)`, Snappy.
- Purpose: Tests dictionary-encoded decimal columns stored as integer physical
  types. Values repeat as `11.11, 11.11, 22.22, 22.22, 33.33, 33.33`.

### `decimal_no_ColumnMetadata.parquet`

- Metadata: empty `created_by`, 10 rows, 1 row group, column
  `_c0: INT64 Decimal(9,1)`, Snappy.
- Purpose: Regression fixture for missing `ColumnMetaData` and
  `rowGroups[0].columns[0].file_offset == 0`; verifies row-group filtering does
  not fail offset checks.

### `tagged_long.parquet`

- Metadata: empty `created_by`, 4 rows, 1 row group, column
  `_c0: INT64 Int(bitWidth=64, isSigned=true)`, Snappy.
- Purpose: Tests reading annotated 64-bit integer logical type as Velox BIGINT.

### `tmmillis_i64.parquet`

- Metadata: `created_by=parquet-mr version 1.8.2-SNAPSHOT`, 3 rows, 2 row
  groups, column `time: INT64 TIMESTAMP_MILLIS`, Snappy.
- Purpose: Tests converted-type timestamp millis stored in INT64 and read into
  Velox timestamp semantics.

### `zero_offset_row_group.parquet`

- Metadata: empty `created_by`, 1 row, 1 row group, column
  `IDX: INT32 INT_32`, uncompressed.
- Purpose: Regression fixture for row-group filtering when a row-group column
  file offset is zero.

## Type conversion and logical annotation fixtures

### `types.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 1 row, 1 row group, 17 leaf
  columns, Snappy. It includes signed integer annotations, float/double,
  `Decimal(38,18)`, UTF8 string, raw binary, boolean, DATE, INT96 timestamp,
  array, struct, and map leaves.
- Purpose: Broad type-coverage fixture for scalar and complex Parquet type
  parsing. Table-scan tests also use it for map reads.

### `enum_type.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group,
  column `test: BYTE_ARRAY Enum`, Gzip.
- Purpose: Tests ENUM converted type mapping to Velox VARCHAR. Expected values
  are `FOO, BAR, FOO`.

### `varbinary_flba.parquet`

- Metadata: `created_by=parquet-mr version 1.13.2-SNAPSHOT`, 100 rows, 1 row
  group, 8 required columns including `binary_field: BYTE_ARRAY`,
  `flba_field: FIXED_LEN_BYTE_ARRAY`, and `int96_field: INT96`, Gzip.
- Purpose: Tests reading FIXED_LEN_BYTE_ARRAY as VARBINARY and validates mixed
  primitive physical type handling.

### `uuid.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group,
  required `uuid_field: FIXED_LEN_BYTE_ARRAY`, Snappy, with no UUID logical
  annotation.
- Purpose: Tests reading fixed-length binary UUID bytes as VARCHAR when the
  requested file schema asks for VARCHAR.

### `upper.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 2 rows, 1 row group,
  required columns `A: INT64` and `b: INT64`, Snappy.
- Purpose: Tests case-normalized field-name reading via
  `fileColumnNamesReadAsLowerCase`.

### `upper_complex.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 1 row, 1 row group, leaves
  `CcLong0: INT64`, map `key: BYTE_ARRAY String`, and nested `CcInt3: INT32`,
  Snappy.
- Purpose: Tests lower-casing of nested field names in a complex schema:
  `Cc` struct containing a map whose value contains an array of structs.

## Array, map, and struct shape fixtures

### `array_0.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 5 rows, 1 row group,
  leaf `element: BYTE_ARRAY String`, definition level 2, repetition level 1,
  Gzip.
- Parquet schema:

  ```text
  message spark_schema {
    optional group _1 (LIST) {
      repeated group list {
        required binary element (STRING);
      }
    }
  }
  ```

- Purpose: Optional array with required elements. Test data covers non-empty
  arrays, an empty array, and a null array.

### `array_1.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group,
  leaf `element: BYTE_ARRAY String`, definition level 1, repetition level 1,
  Gzip.
- Parquet schema:

  ```text
  message spark_schema {
    required group _1 (LIST) {
      repeated group list {
        required binary element (STRING);
      }
    }
  }
  ```

- Purpose: Required array with required elements, including an empty array.

### `array_2.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group,
  leaf `element: BYTE_ARRAY String`, definition level 2, repetition level 1,
  Gzip.
- Parquet schema:

  ```text
  message spark_schema {
    required group _1 (LIST) {
      repeated group list {
        optional binary element (STRING);
      }
    }
  }
  ```

- Purpose: Required array with optional elements, including null array elements.

### `array_3.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group,
  leaf `element: BYTE_ARRAY String`, definition level 1, repetition level 1,
  Gzip.
- Parquet schema:

  ```text
  message spark_schema {
    repeated binary element (STRING);
  }
  ```

- Purpose: Legacy required-array encoding where the repeated field itself is
  named `element`.

### `array_of_array1.parquet`

- Metadata: `created_by=parquet-cpp version 1.5.1-SNAPSHOT`, 2 rows, 1 row
  group, leaf `item: BYTE_ARRAY String`, definition level 5, repetition level 2,
  Snappy.
- Parquet schema:

  ```text
  message schema {
    optional group _1 (LIST) {
      repeated group list {
        optional group item (LIST) {
          repeated group list {
            optional binary item (STRING);
          }
        }
      }
    }
  }
  ```

- Purpose: Nested array fixture. Test data includes a null outer value and an
  outer array containing a non-empty inner array and a null inner array.

### `old_repeated_int.parquet`

- Metadata: `created_by=parquet-mr`, 1 row, 1 row group,
  leaf `repeatedInt: INT32`, definition level 1, repetition level 1,
  uncompressed.
- Parquet schema:

  ```text
  message TestProtobuf.RepeatedIntMessage {
    repeated int32 repeatedInt;
  }
  ```

- Purpose: Legacy repeated primitive encoding read as `ARRAY(INTEGER)`;
  also verifies an error is raised if requested as scalar INTEGER.

### `proto_repeated_string.parquet`

- Metadata: `created_by=parquet-mr`, 3 rows, 1 row group,
  leaf `myString: BYTE_ARRAY String`, definition level 1, repetition level 1,
  uncompressed.
- Parquet schema:

  ```text
  message TestProtobuf.StringArray {
    repeated binary myString (STRING);
  }
  ```

- Purpose: Protobuf-style repeated string field read as an array of strings.

### `nested_array_struct.parquet`

- Metadata: `created_by=parquet-mr`, 3 rows, 1 row group, leaves
  `primitive: INT32`, `id: INT32`, and nested `someId: INT32`, uncompressed.
- Parquet schema:

  ```text
  message TestProtobuf.ArrayWithNestedGroupAndArray {
    optional int32 primitive;
    repeated group myComplex {
      optional int32 id;
      repeated group repeatedMessage {
        optional int32 someId;
      }
    }
  }
  ```

- Purpose: Tests name-based mapping through an array of structs with a nested
  repeated-message field.

### `struct_of_array.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 3 rows, 1 row group, leaves
  `element: BYTE_ARRAY String` and `element: INT32`, both definition level 4 and
  repetition level 1, Snappy.
- Parquet schema:

  ```text
  message spark_schema {
    optional group struct {
      optional group a0 (LIST) {
        repeated group list {
          optional binary element (STRING);
        }
      }
      optional group a1 (LIST) {
        repeated group list {
          optional int32 element;
        }
      }
    }
  }
  ```

- Purpose: Struct containing arrays; used by table-scan filter tests that filter
  on nested array fields and expect null struct handling.

### `struct_of_array_of_array.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 13,520 rows, 1 row group,
  leaves `array: BYTE_ARRAY String` and `array: INT32`, definition level 4,
  repetition level 2, Gzip.
- Parquet schema:

  ```text
  message hive_schema {
    optional group test {
      optional group stringarrayfield (LIST) {
        repeated group array (LIST) {
          repeated binary array (STRING);
        }
      }
      optional group intarrayfield (LIST) {
        repeated group array (LIST) {
          repeated int32 array;
        }
      }
    }
  }
  ```

- Purpose: Struct with two array-of-array children. Verifies schema inference
  for nested repeated groups and reading deeply nested arrays.

### `single_row_struct.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 1 row, 1 row group,
  required leaves `a: INT64` and `b: INT64`, Snappy.
- Parquet schema:

  ```text
  message spark_schema {
    required group s {
      required int64 a;
      required int64 b;
    }
  }
  ```

- Purpose: Minimal struct fixture used to verify a single row struct scan.

### `row_map_array.parquet`

- Metadata: `parquet-tools inspect` fails with
  `ArrowInvalid: Map keys must be annotated as required`.
- Parquet schema:

  ```text
  message schema {
    optional group c {
      optional int64 c0;
      optional group c1 (MAP) {
        repeated group map {
          optional binary key (STRING);
          optional group value (LIST) {
            repeated group list {
              optional int32 element;
            }
          }
        }
      }
    }
  }
  ```

- Purpose: Legacy/invalid map fixture that Velox still supports. The intended
  schema is one row of `ROW(c ROW(c0 BIGINT, c1 MAP(VARCHAR, ARRAY(INTEGER))))`.

### `map_key_value.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 1 row, 1 row group,
  leaves `key: BYTE_ARRAY String` and `value: INT64`, definition levels 2 and 3,
  repetition level 1, Gzip.
- Parquet schema:

  ```text
  message hive_schema {
    optional group test (MAP) {
      repeated group map {
        required binary key (STRING);
        optional int64 value;
      }
    }
  }
  ```

- Purpose: Older map encoding that uses `MAP_KEY_VALUE` instead of `MAP` as the
  map SchemaElement converted type. Contains one map with 8 entries.

### `unnotated_map.parquet`

- Metadata: `created_by=parquet-mr version 1.11.1-uber-0.10`, 4 rows, 1 row
  group, leaves `key: BYTE_ARRAY String` and `value: INT64`, Gzip.
- Parquet schema:

  ```text
  message hive_schema {
    optional group test (MAP) {
      repeated group key_value (MAP_KEY_VALUE) {
        required binary key (STRING);
        optional int64 value;
      }
    }
  }
  ```

- Purpose: Legacy map fixture, intentionally named `unnotated_map.parquet` in
  the repository, with a `MAP_KEY_VALUE` node used to verify old Hive-style map
  parsing.

### `unannotated_list.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 22 rows, 1 row group,
  leaves `a: INT64`, `b: BOOLEAN`, `c: BYTE_ARRAY String`, Gzip.
- Parquet schema:

  ```text
  message ParquetSchema {
    optional group self (LIST) {
      repeated group self_tuple {
        optional int64 a;
        optional boolean b;
        required binary c (STRING);
      }
    }
  }
  ```

- Purpose: LIST-annotated group without the normal middle `list` layer:
  `optional group self (LIST) { repeated group self_tuple { ... } }`. Velox
  creates a dummy row layer for the tuple fields.

### `listmultiplechildren.parquet`

- Metadata: `created_by=parquet-mr`, 18,033 rows, 1 row group, leaves
  `a: INT64`, `b: BOOLEAN`, `c: BYTE_ARRAY String`, Snappy.
- Parquet schema:

  ```text
  message hive_schema {
    optional group test (LIST) {
      repeated group array {
        optional int64 a;
        optional boolean b;
        optional binary c (STRING);
      }
    }
  }
  ```

- Purpose: Legacy LIST fixture whose repeated `array` node has more than one
  child. Velox must interpret array elements as rows.

### `parse_empty_nested_list.parquet`

- Metadata: `created_by=parquet-cpp-arrow version 19.0.1`, 1000 rows, 1 row
  group, leaf `element: INT32`, definition level 5, repetition level 1, Snappy.
- Parquet schema:

  ```text
  message schema {
    optional group msg {
      optional group a {
        optional group b (LIST) {
          repeated group list {
            optional int32 element;
          }
        }
      }
    }
  }
  ```

- Purpose: Tests a nested structure where every row is
  `{ msg: { a: { b: [] } } }`. Preserves empty nested list shape across 1000
  rows.

### `array_of_row_hive_reserved_keywords.parquet`

- Metadata: `created_by=parquet-mr version 1.13.1`, 6 rows, 1 row group, leaves
  `id: INT32`, `name: BYTE_ARRAY String`, `quantity: INT32`, `price: DOUBLE`,
  Gzip.
- Parquet schema:

  ```text
  message hive_schema {
    optional int32 id;
    optional group items (LIST) {
      repeated group bag {
        optional group array_element {
          optional binary name (STRING);
          optional int32 quantity;
          optional double price;
        }
      }
    }
  }
  ```

- Purpose: Hive writer fixture where an `ARRAY<ROW<...>>` uses reserved field
  names such as `array_element`. Verifies Velox reconstructs
  `ROW<id:INTEGER, items:ARRAY<ROW<name, quantity, price>>>`.

### `array_of_map_of_int_key_array_value.parquet`

- Parquet schema:

  ```text
  message hive_schema {
    optional group test (LIST) {
      repeated group array (MAP) {
        repeated group key_value (MAP_KEY_VALUE) {
          required binary key (UTF8);
          optional group value (LIST) {
            repeated int32 array;
          }
        }
      }
    }
  }
  ```

- Purpose: Legacy nested schema:
  `ARRAY<MAP<VARCHAR, ARRAY<INTEGER>>>`. It verifies Velox accepts a repeated
  `array` group inside a MAP value even though pyarrow rejects the layout.

### `array_of_map_of_int_key_struct_value.parquet`

- Parquet schema:

  ```text
  message hive_schema {
    optional group test (LIST) {
      repeated group array (MAP) {
        repeated group key_value (MAP_KEY_VALUE) {
          required int32 key;
          optional group value {
            optional binary stringfield (UTF8);
            optional int64 longfield;
          }
        }
      }
    }
  }
  ```

- Purpose: Legacy nested schema:
  `ARRAY<MAP<INTEGER, ROW<stringfield VARCHAR, longfield BIGINT>>>`. It covers
  map values that are structs inside an array.

### `proto-struct-with-array.parquet`

- Metadata: `created_by=parquet-mr`, 1 row, 1 row group, 6 leaves:
  optional/required primitive INT32, repeated primitive INT32, optional/required
  struct field `someId`, and repeated struct field `someId`, uncompressed.
- Parquet schema:

  ```text
  message TestProtobuf.SchemaConverterRepetition {
    optional int32 optionalPrimitive;
    required int32 requiredPrimitive;
    repeated int32 repeatedPrimitive;
    optional group optionalMessage {
      optional int32 someId;
    }
    required group requiredMessage {
      optional int32 someId;
    }
    repeated group repeatedMessage {
      optional int32 someId;
    }
  }
  ```

- Purpose: Protobuf-style optional, required, and repeated fields. Tests
  repeated primitive arrays and arrays of structs in the same file.

### `null_map.parquet`

- Metadata: `created_by=parquet-mr version 1.12.2`, 1 row, 1 row group,
  leaves `i: BYTE_ARRAY String`, map `key: BYTE_ARRAY String`, and
  `value: BYTE_ARRAY String`, Snappy.
- Parquet schema:

  ```text
  message spark_schema {
    required binary i (STRING);
    optional group c (MAP) {
      repeated group key_value {
        required binary key (STRING);
        optional binary value (STRING);
      }
    }
  }
  ```

- Purpose: Verifies reading a null map value alongside a scalar string column.

## Page, encoding, compression, and corruption fixtures

### `v2_page.parquet`

- Metadata: empty `created_by`, 5 rows, 1 row group, column `regionkey: INT64`,
  definition level 1, repetition level 0, uncompressed.
- Purpose: Data page v2 fixture with zero max repetition level. Expected values
  are `0..4`.

### `complex_type_v2_page.parquet`

- Metadata: `created_by=parquet-mr version 1.13.1`, 1 row, 1 row group, leaves
  for `ARRAY(INTEGER)` and `MAP(VARCHAR, INTEGER)`, Snappy.
- Purpose: Data page v2 fixture for complex types. Expected data is
  `nums = [4, 5]` and `props = {"x": 99, "y": 100}`.

### `empty_v2datapage.parquet`

- Metadata: `created_by=parquet-mr`, 30,001 rows, 1 row group,
  column `test: FLOAT`, Snappy, serialized size 94.
- Purpose: Regression fixture for an empty data page v2 layout. The expected
  column is all nulls when read as REAL.

### `delta_byte_array.parquet`

- Metadata: `created_by=parquet-mr version 1.13.1`, 4 rows, 1 row group,
  required `a: BYTE_ARRAY String`, uncompressed.
- Purpose: Tests DELTA_BYTE_ARRAY decoding. Expected strings are
  `axis, axle, babble, babyhood`.

### `parquet-251.parquet`

- Metadata: `created_by=parquet-mr version 1.6.0`, 5 rows, 1 row group,
  columns `id: INT32` and `str: BYTE_ARRAY String`, uncompressed.
- Purpose: Regression fixture for filtering string values from an older
  parquet-mr writer. The test filters `str = '2'`.

### `lzo.parquet`

- Metadata: `created_by=parquet-mr`, 23,547 rows, 1 row group, leaves
  `intfield: INT32` and `array_element: BYTE_ARRAY String`, LZO compression.
- Purpose: Verifies LZO data page decompression and nested array reads.

### `small_page_header`

- Metadata: serialized page-header bytes, not a complete Parquet file, 126
  bytes.
- Purpose: Unit fixture for `PageReader::readPageHeader`. Expected DATA_PAGE
  header has uncompressed size 16,950, compressed size 10,759 and 21,738 values,
  min value of 39 `a` characters, and max value of 49 `z` characters.

### `large_page_header`

- Metadata: serialized page-header bytes, not a complete Parquet file, 3,590
  bytes.
- Purpose: Unit fixture for a large page header. Expected DATA_PAGE header has
  uncompressed size 1,050,822, compressed size 66,759 and 970 values, min value of
  1,295 `a` characters, and max value of 2,255 `z` characters.

### `corrupted_page_header`

- Metadata: intentionally corrupted serialized page-header bytes, not a
  complete Parquet file, 3,520 bytes.
- Purpose: Negative unit fixture. The encoded `min_value` length is wrong on
  purpose so `PageReader::readPageHeader` must throw a Velox exception.

## External table and compatibility fixtures

### `icebergNullIcebergPartition.parquet`

- Metadata: `created_by=parquet-mr version 1.13.1`, 2 rows, 1 row group,
  columns `c0: INT64` required and `c1: INT64` optional, ZSTD.
- Purpose: Copied from a null partition in an Iceberg v2 table written by
  Spark. Tests partition-key null filtering where file data is paired with a
  split partition value of null.

### `complex_with_varchar_varbinary.parquet`

- Metadata: `created_by=parquet-cpp-velox`, 1 row, 1 row group, 6 leaves:
  arrays of `BYTE_ARRAY String` and raw `BYTE_ARRAY`, maps with string and raw
  binary keys, and BIGINT values, Gzip.
- Purpose: Tests requested schemas that distinguish VARCHAR and VARBINARY inside
  arrays and maps, and verifies dictionary-wrapped nested values.
