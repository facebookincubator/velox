# Column Extraction Pushdown Proposal

*2026-03-04*

## Motivation

Queries like `SELECT map_keys(col) FROM t`, `SELECT cardinality(col) FROM t`,
or `SELECT map_values(col).x FROM t` currently read the entire complex-typed
column from storage, even though only a subset of its physical streams is
needed.  For a `MAP(K, V)` column, `map_keys` only needs keys (not values), and
`cardinality` only needs the lengths stream (neither keys nor values).

This waste is significant for wide map/struct values — the reader
materializes data that the engine immediately discards.

**Column extraction pushdown** is a new pushdown mechanism, separate from
subfield pruning, that tells the reader to extract a specific component from a
complex type and produce it as a **different output type**.  For example,
extracting map keys produces an `ARRAY(K)` instead of a `MAP(K, V)`.

### Distinction from subfield pruning

|                | Subfield pruning                     | Column extraction                                  |
|----------------|--------------------------------------|----------------------------------------------------|
| Output type    | Same as input (MAP stays MAP)        | Different (MAP -> ARRAY, MAP/ARRAY -> BIGINT)      |
| Semantics      | Drop unused parts, null-fill         | Transform the type structure                       |
| Mechanism      | `requiredSubfields` on column handle | `extractions` (NamedExtraction list) on column handle |
| Composability  | N/A                                  | N/A — mutually exclusive with subfield pruning     |

### Why not extend subfield pruning?

Consider this query:

```sql
SELECT cardinality(col) AS a, col['foo'] AS b FROM t
```

where `col` is `MAP(VARCHAR, BIGINT)`.

Both `a` and `b` reference the same column `col`, but they need fundamentally
different things:

- `a` needs only the size — no keys, no values.
- `b` needs the key `"foo"` and its corresponding value.

With subfield pruning, a single column handle represents `col`.  The required
subfields from both references are merged:

```
requiredSubfields: ["col[$]", "col[\"foo\"]"]
```

These are contradictory: `[$]` says "skip keys and values" while `["foo"]` says
"read key `foo` and its value." The reader cannot satisfy both with one pass —
it must read the whole column in order to return maps with correct size, and
if it only reads key `foo` the result would be incorrect.

Column extraction solves this with a single column handle that carries
**multiple named extraction chains**.  The column is read once, and each
extraction is applied independently:

```
assignments: {
    "col": {
        name: "col",
        hiveType: MAP(VARCHAR, BIGINT),
        dataType: ROW({ "a": BIGINT, "b": ARRAY(BIGINT) }),
        extractions: [
            { outputName: "a", chain: [Size],  dataType: BIGINT },
            { outputName: "b", chain: [MapKeyFilter(["foo"]), MapValues],  dataType: ARRAY(BIGINT) }
        ],
    }
}
```

The reader reads `col` once — it reads sizes without restriction (`a`), and read
keys and values with a filter on the key (since `b` needs only key `"foo"`), and
further apply remaining of the chain in `b` in the column reader of map values.

This is the core reason column extraction is a new mechanism rather than an
extension of subfield pruning.  Subfield pruning merges all references to a
column into one output, which forces the reader to satisfy the union of all
requirements.  Column extraction keeps a single column handle but produces
multiple named outputs, each with its own extraction chain.

## Protocol: Extraction Chain

The extraction is expressed as an ordered chain of steps.  Each step operates on
one nesting level of the source type.  The chain is a new field on the column
handle, mutually exclusive with `requiredSubfields`.

### Extraction steps

```
ExtractionStep =
  | StructField(name: string)    // Navigate into a struct field.
  | MapKeys                      // Extract map keys.  MAP(K, V) → ARRAY(K).
  | MapValues                    // Extract map values.  MAP(K, V) → ARRAY(V).
  | MapKeyFilter(keys: list)     // Filter map to specific keys.
  |                              //   MAP(K, V) → MAP(K, V).  Type-preserving.
  |                              //   Keys are strings or integers depending on
  |                              //   the map's key type.
  | ArrayElements                // Navigate into array elements.
  | Size                         // Extract size.  MAP/ARRAY → BIGINT.  Terminal.
```

### Step input/output types (validation)

Each step's output feeds as input to the next step.  This forms a linear
pipeline for **type validation**:

| Step               | Required input              | Output       |
|--------------------|-----------------------------|--------------|
| `StructField(f)`   | `ROW(..., f: T, ...)`       | `T`          |
| `MapKeys`          | `MAP(K, V)`                 | `ARRAY(K)`   |
| `MapValues`        | `MAP(K, V)`                 | `ARRAY(V)`   |
| `MapKeyFilter(ks)` | `MAP(K, V)`                 | `MAP(K, V)`  |
| `ArrayElements`    | `ARRAY(T)`                  | `T`          |
| `Size`             | `MAP(K, V)` or `ARRAY(T)`   | `BIGINT`     |

**Rule:** `MapKeys` and `MapValues` produce `ARRAY(...)`.  Any subsequent step
sees `ARRAY` as input.  Therefore, `MapKeys`/`MapValues` **must** be followed
by `ArrayElements` unless it is the last step in the chain.  `MapKeyFilter` is
type-preserving (`MAP` → `MAP`) and does NOT require `ArrayElements` after it.

### Output type derivation

The validation pipeline loses nesting information (each `ArrayElements`
unwraps one `ARRAY` layer).  The actual output type is derived recursively:

```
derive(T, []) = T

derive(ROW(.., f:T, ..), [StructField(f),               ...rest]) = derive(T, rest)

derive(MAP(K, V),         [MapKeys,       ArrayElements, ...rest]) = ARRAY(derive(K, rest))
derive(MAP(K, V),         [MapKeys])                               = ARRAY(K)

derive(MAP(K, V),         [MapValues,     ArrayElements, ...rest]) = ARRAY(derive(V, rest))
derive(MAP(K, V),         [MapValues])                             = ARRAY(V)

derive(MAP(K, V),         [MapKeyFilter,                ...rest]) = derive(MAP(K, V), rest)

derive(ARRAY(T),          [ArrayElements,                ...rest]) = ARRAY(derive(T, rest))

derive(MAP|ARRAY,         [Size])                                  = BIGINT
```

`MapKeys`/`MapValues` + `ArrayElements` are consumed as a pair — the
extraction wraps in `ARRAY`, and `ArrayElements` enters it.  `MapKeyFilter` is
type-preserving and does not consume an `ArrayElements`.  A standalone
`ArrayElements` (without a preceding extraction step) handles a source `ARRAY`
level.

### `ArrayElements` roles

| Context                      | Role                                                                                          |
|------------------------------|-----------------------------------------------------------------------------------------------|
| After `MapKeys`/`MapValues`  | Consumes the `ARRAY` produced by extraction.  **Mandatory** to continue the chain.            |
| On a source `ARRAY` type     | Navigates into the array's elements.  **Mandatory** when the source type has an `ARRAY` level.|

Every `ARRAY` boundary in the chain — whether from extraction or from the
source type — is explicitly represented by an `ArrayElements` step.

### Mutual exclusivity with subfield pruning

Column extraction (`extractions`) and subfield pruning (`requiredSubfields`)
**cannot coexist** on the same column handle.  A column handle uses either
`requiredSubfields` (type-preserving pruning) or `extractions` (extraction
chains), but not both.

This simplifies the reader contract — the reader either prunes subfields
within the existing type structure, or applies extraction chains to produce
new types.  There is no ambiguity about which operation runs first or how
their type changes interact.

**When to use which:**

- Use `requiredSubfields` when the column's output type stays the same as the
  file type (struct field pruning, map key filtering, array index truncation).
- Use `extractions` when the column's output type changes (MapKeys, MapValues,
  Size, StructField extraction, or any chain that transforms the type).

**Expressing subfield-like operations in extraction chains:**

Operations that would have used `requiredSubfields` can be expressed as
extraction steps instead:

- Struct field access → `StructField(name)` in the chain
- Map key filtering → `MapKeyFilter(keys)` in the chain
- Multiple struct fields → multiple `NamedExtraction` entries, each with a
  `StructField` chain

For example, instead of `requiredSubfields: ["col.x", "col.y"]`, use:
```
extractions: [
    { outputName: "x", chain: [StructField("x")], dataType: INT },
    { outputName: "y", chain: [StructField("y")], dataType: INT }
]
```

### Overlap between `MapKeyFilter` and map key pruning

`MapKeyFilter` and subfield pruning's map key subscripts (`["key1"]`, `["key2"]`)
both filter a map to specific keys.  They overlap in functionality but differ in
where they sit:

- **Subfield pruning** (`requiredSubfields: ["col[\"foo\"]"]`): filters keys on
  the source column, output stays `MAP(K, V)`.  No extraction chain involved.
- **`MapKeyFilter`**: filters keys as a step in the extraction chain, output
  stays `MAP(K, V)`.  Can compose with other extraction steps.

Since extraction and subfield pruning are mutually exclusive on the same column
handle, use one or the other:

- Use `requiredSubfields` when key filtering is the only operation needed and
  no type transformation is involved.
- Use `MapKeyFilter` in an extraction chain when key filtering combines with
  other extraction steps (e.g., `[MapKeyFilter(["foo"]), MapValues]`).

### Overlap between `StructField` and struct subfield pruning

`StructField` and subfield pruning's nested field paths (`.field1`, `.field2`)
both navigate into struct fields.  They overlap when accessing struct children:

- **Subfield pruning** (`requiredSubfields: ["col.x"]`): prunes the struct to
  only field `x`, output stays `ROW(...)` with other fields null-filled.
- **`StructField("x")`**: extracts field `x` as a step in the extraction chain,
  output becomes the field's type directly (e.g., `INT`).

Since extraction and subfield pruning are mutually exclusive:

- Use `requiredSubfields` for struct pruning when no type transformation is
  needed (output stays ROW).
- Use `StructField` in an extraction chain when struct navigation combines
  with other extraction steps (e.g., `[StructField("a"), MapKeys]`).

## Examples

### `map_keys(col)` — `col: MAP(K, V)`

```
Chain: [MapKeys]
Validation:  MAP(K,V) → MapKeys → ARRAY(K) ✓
Output:      ARRAY(K)
```

### `map_keys(col.a.b)` — `col: ROW(a: ROW(b: MAP(K, V)))`

```
Chain: [StructField("a"), StructField("b"), MapKeys]
Validation:  ROW → StructField("a") → ROW → StructField("b") → MAP(K,V) → MapKeys → ARRAY(K) ✓
Output:      ARRAY(K)
```

### `cardinality(col)` — `col: MAP(K, V)` or `ARRAY(T)`

```
Chain: [Size]
Validation:  MAP(K,V) → Size → BIGINT ✓
Output:      BIGINT
```

### `col.x` — `col: ROW(x: INT, y: INT)`

Struct field access with no other extraction steps.  Prefer subfield pruning
(`requiredSubfields: ["col.x"]`) which keeps the `ROW` type and uses existing
infrastructure.  `StructField` is only needed when it is part of a larger
extraction chain (see `map_keys(col.a.b)` and `cardinality(col.features)`).

```
-- Preferred: subfield pruning
extraction:          []
requiredSubfields:   ["col.x"]

-- Also valid but unnecessary: extraction
Chain: [StructField("x")]
Output:      INT
```

### `cardinality(col.features)` — `col: ROW(features: ARRAY(FLOAT), label: INT)`

Size extraction on a nested field.  Navigate into the struct, then extract
size.

```
Chain: [StructField("features"), Size]
Validation:  ROW → StructField("features") → ARRAY(FLOAT) → Size → BIGINT ✓
Output:      BIGINT
```

### `map_keys(map_values(col))` — `col: MAP(K1, MAP(K2, V))`

This represents `transform(map_values(col), x -> map_keys(x))` — for each
value (which is `MAP(K2, V)`), extract its keys.

```
Chain: [MapValues, ArrayElements, MapKeys]
Validation:  MAP(K1, MAP(K2,V)) → MapValues → ARRAY(MAP(K2,V))
                                 → ArrayElements → MAP(K2,V)
                                 → MapKeys → ARRAY(K2)  ✓
Output:      derive(MAP(K1, MAP(K2,V)), [MV, AE, MK])
             = ARRAY(derive(MAP(K2,V), [MK]))
             = ARRAY(ARRAY(K2))
```

### `map_keys(array_elements(map_values(col)))` — `col: MAP(K1, ARRAY(MAP(K2, V)))`

```
Chain: [MapValues, ArrayElements, ArrayElements, MapKeys]
Validation:  MAP(K1, ARRAY(MAP(K2,V))) → MapValues   → ARRAY(ARRAY(MAP(K2,V)))
                                        → ArrayElements → ARRAY(MAP(K2,V))
                                        → ArrayElements → MAP(K2,V)
                                        → MapKeys       → ARRAY(K2)  ✓
Output:      derive(MAP(K1, ARRAY(MAP(K2,V))), [MV, AE, AE, MK])
             = ARRAY(derive(ARRAY(MAP(K2,V)), [AE, MK]))
             = ARRAY(ARRAY(derive(MAP(K2,V), [MK])))
             = ARRAY(ARRAY(ARRAY(K2)))
```

Two `ArrayElements` — the first consumes the `ARRAY` from `MapValues`, the
second navigates the source `ARRAY` level.

### `map_values(col).x` — `col: MAP(K, ROW(x: INT, y: INT))`

Extraction is used, so `requiredSubfields` cannot be set.  Use `StructField`
in the chain to extract the specific field.

```
Chain: [MapValues, ArrayElements, StructField("x")]
Validation:  MAP(K, ROW(x,y)) → MapValues     → ARRAY(ROW(x,y))
                               → ArrayElements → ROW(x,y)
                               → StructField   → INT ✓
Output:      ARRAY(INT)
```

### `map_subset(col, ARRAY['a', 'b'])` — `col: MAP(VARCHAR, BIGINT)`

Map key filtering with no other extraction steps.  Prefer subfield pruning
(`requiredSubfields: ["col[\"a\"]", "col[\"b\"]"]`) which uses existing
infrastructure.  `MapKeyFilter` is only needed when it is part of a larger
extraction chain (see the nested key filter example below).

```
-- Preferred: subfield pruning
extraction:          []
requiredSubfields:   ["col[\"a\"]", "col[\"b\"]"]

-- Also valid but unnecessary: extraction
Chain: [MapKeyFilter(["a", "b"])]
Output:      MAP(VARCHAR, BIGINT)
```

### `element_at(col, 'foo').x` — `col: MAP(VARCHAR, ROW(x: INT, y: INT))`

Single-key filter with single struct field access.  Since `MapKeyFilter` is
the only extraction step, prefer subfield pruning:

```
-- Preferred: subfield pruning (no extraction chain)
extraction:          []
requiredSubfields:   ["col[\"foo\"].x"]

-- Also valid: extraction chain avoids materializing the full ROW, which can
-- be beneficial when the struct has many fields.
Chain: [MapKeyFilter(["foo"]), MapValues, ArrayElements, StructField("x")]
Output:      ARRAY(INT)
```

### Non-pushable: `map_keys(map_filter(col, (k, v) -> v > 10))` — `col: MAP(VARCHAR, BIGINT)`

The `map_filter` predicate depends on values (`v > 10`).  If we pushed
`MapKeys` extraction to the reader, the reader would skip values, making it
impossible to evaluate the filter.  **Extraction cannot be pushed** through
intermediate expressions that depend on skipped data.  The entire map must be
read, `map_filter` applied in the engine, and then `map_keys` applied to the
filtered result.

### Nested key filter — `col: MAP(K1, MAP(VARCHAR, ROW(x: INT, y: INT)))`

SQL: `transform(map_values(col), m -> element_at(m, 'foo').x)`

Extract values from outer map, filter inner map to key `"foo"`, extract
subfield `x`.  Extraction is used, so `requiredSubfields` cannot be set.
Use `StructField` in the chain.

```
Chain: [MapValues, ArrayElements, MapKeyFilter(["foo"]), MapValues, ArrayElements, StructField("x")]
Validation:  MAP(K1, MAP(VARCHAR, ROW(x,y))) → MapValues     → ARRAY(MAP(VARCHAR, ROW(x,y)))
                                              → ArrayElements → MAP(VARCHAR, ROW(x,y))
                                              → MapKeyFilter  → MAP(VARCHAR, ROW(x,y))
                                              → MapValues     → ARRAY(ROW(x,y))
                                              → ArrayElements → ROW(x,y)
                                              → StructField   → INT ✓
Output:      ARRAY(ARRAY(INT))
```

### Error: missing `ArrayElements` after `MapValues`

```
Chain: [MapValues, MapKeys]   on MAP(K1, ARRAY(MAP(K2, V)))
Validation:  MAP(K1, ARRAY(MAP(K2,V))) → MapValues → ARRAY(ARRAY(MAP(K2,V)))
                                        → MapKeys → ERROR: expects MAP, got ARRAY
```

### Error: missing `ArrayElements` after `MapKeys`

```
Chain: [MapKeys, StructField("x")]   on MAP(ROW(x: INT, y: INT), V)
Validation:  MAP(ROW(x,y), V) → MapKeys → ARRAY(ROW(x,y))
                               → StructField("x") → ERROR: expects ROW, got ARRAY
```

## Column Handle Protocol

### `HiveColumnHandle` (C++)

```cpp
/// Type of extraction to apply at one nesting level.
enum class ExtractionStep : uint8_t {
  /// Navigate into a struct field.  Input must be ROW.
  kStructField,
  /// Extract map keys as ARRAY.  Input must be MAP.
  kMapKeys,
  /// Extract map values as ARRAY.  Input must be MAP.
  kMapValues,
  /// Filter map to specific keys.  Input must be MAP.  Type-preserving.
  kMapKeyFilter,
  /// Navigate into array elements.  Input must be ARRAY.
  kArrayElements,
  /// Extract size as BIGINT.  Input must be MAP or ARRAY.  Terminal.
  kSize,
};

/// One step in the extraction chain.
struct ExtractionPathElement {
  ExtractionStep step;

  /// Field name, only used when step == kStructField.
  std::string fieldName;

  /// Filter keys, only used when step == kMapKeyFilter.
  /// Contains the set of map keys to retain.  Use stringFilterKeys for
  /// VARCHAR keys or intFilterKeys for BIGINT/INTEGER keys.
  std::vector<std::string> stringFilterKeys;
  std::vector<int64_t> intFilterKeys;
};

/// Named extraction chain producing one output column.
struct NamedExtraction {
  /// Output column name in the scan's outputType.
  std::string outputName;

  /// Extraction chain to apply.  Empty means pass-through (no extraction).
  std::vector<ExtractionPathElement> chain;

  /// Output type after applying the chain.
  TypePtr dataType;
};

class HiveColumnHandle : public connector::ColumnHandle {
 public:
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<NamedExtraction> extractions = {},
      ...);

  /// Named extraction chains.  Empty means no extraction (current behavior).
  /// When a single entry is present, the column handle's dataType is that
  /// entry's dataType.  When multiple entries are present, the column
  /// handle's dataType is a ROW type whose fields are the outputNames with
  /// their corresponding dataTypes.
  /// Mutually exclusive with requiredSubfields — if extractions is non-empty,
  /// requiredSubfields must be empty.
  const std::vector<NamedExtraction>& extractions() const {
    return extractions_;
  }

 private:
  ...
  std::vector<NamedExtraction> extractions_;
};
```

### `HiveColumnHandle` (Java / Presto coordinator)

```java
public class HiveColumnHandle extends BaseHiveColumnHandle {
    ...
    // Existing
    private final List<Subfield> requiredSubfields;

    // New: named extraction chains
    private final List<NamedExtraction> extractions;

    public record NamedExtraction(
        String outputName,
        List<ExtractionPathElement> chain,
        TypeSignature dataType
    ) {}

    public record ExtractionPathElement(
        ExtractionStep step,
        Optional<String> fieldName,           // only for STRUCT_FIELD
        Optional<List<String>> stringFilterKeys, // only for MAP_KEY_FILTER (VARCHAR keys)
        Optional<List<Long>> intFilterKeys    // only for MAP_KEY_FILTER (BIGINT keys)
    ) {}

    public enum ExtractionStep {
        STRUCT_FIELD,
        MAP_KEYS,
        MAP_VALUES,
        MAP_KEY_FILTER,
        ARRAY_ELEMENTS,
        SIZE
    }
}
```

### Serialization

The named extraction chains are serialized as a JSON array in the plan
fragment.  Note that `requiredSubfields` and `extractions` are mutually
exclusive — when `extractions` is non-empty, `requiredSubfields` must be
empty:

```json
{
  "name": "col",
  "hiveType": "map(varchar, array(map(integer, double)))",
  "requiredSubfields": [],
  "extractions": [
    {
      "outputName": "col_keys",
      "dataType": "array(array(array(integer)))",
      "chain": [
        {"step": "MAP_VALUES"},
        {"step": "ARRAY_ELEMENTS"},
        {"step": "ARRAY_ELEMENTS"},
        {"step": "MAP_KEYS"}
      ]
    }
  ]
}
```

With key filter and struct field extraction:

```json
{
  "name": "col",
  "hiveType": "map(varchar, row(x integer, y integer))",
  "requiredSubfields": [],
  "extractions": [
    {
      "outputName": "col_x",
      "dataType": "array(integer)",
      "chain": [
        {"step": "MAP_KEY_FILTER", "stringFilterKeys": ["foo", "bar"]},
        {"step": "MAP_VALUES"},
        {"step": "ARRAY_ELEMENTS"},
        {"step": "STRUCT_FIELD", "fieldName": "x"}
      ]
    }
  ]
}
```

Multiple extractions from the same column:

```json
{
  "name": "col",
  "hiveType": "map(varchar, row(x integer, y double))",
  "requiredSubfields": [],
  "extractions": [
    {
      "outputName": "col_size",
      "dataType": "bigint",
      "chain": [{"step": "SIZE"}]
    },
    {
      "outputName": "col_keys",
      "dataType": "array(varchar)",
      "chain": [{"step": "MAP_KEYS"}]
    },
    {
      "outputName": "col_x",
      "dataType": "array(integer)",
      "chain": [
        {"step": "MAP_VALUES"},
        {"step": "ARRAY_ELEMENTS"},
        {"step": "STRUCT_FIELD", "fieldName": "x"}
      ]
    }
  ]
}
```

### Contract

- `extractions` is empty → current behavior, `requiredSubfields` may be used
  for type-preserving pruning.
- `extractions` is non-empty → worker applies each chain, producing the
  corresponding `dataType`.  `requiredSubfields` must be empty.
- `extractions` and `requiredSubfields` are **mutually exclusive** on the same
  column handle.
- `requiredSubfields` operates on the file type (existing behavior).
- When multiple extractions are needed from the same column, use the
  `NamedExtraction` list on a single column handle (see "Multiple extractions
  per column" below).  Do NOT create multiple column handles for the same
  source column.
- The worker validates `hiveType` + each extraction chain → its `dataType`
  and rejects mismatches.

### Multiple extractions per column

When a query references the same column with different extractions (e.g.,
`SELECT map_keys(col) AS keys, cardinality(col) AS size FROM t`), a single
column handle carries all extraction chains via `NamedExtraction`.  The column
handle's `dataType` is a **ROW** whose fields are the output names with their
corresponding types:

```
assignments: {
    "col": HiveColumnHandle {
        name: "col",
        hiveType: MAP(K, V),
        dataType: ROW({ "keys": ARRAY(K), "size": BIGINT }),
        extractions: [
            { outputName: "keys", chain: [MapKeys],  dataType: ARRAY(K) },
            { outputName: "size", chain: [Size],     dataType: BIGINT   }
        ]
    }
}
```

The column is read once from the file.  Each extraction chain is applied
independently to produce a field in the output ROW.

**Examples with multiple extractions:**

```
-- col: MAP(VARCHAR, ROW(x: INT, y: INT, z: INT))
-- Query: SELECT map_keys(col) AS keys, map_values(col).x AS vals_x FROM t
--
-- Two extractions: keys and a specific value subfield.
-- Use StructField in the values chain to extract only field x.

HiveColumnHandle {
    name: "col",
    hiveType: MAP(VARCHAR, ROW(x: INT, y: INT, z: INT)),
    dataType: ROW({ "keys": ARRAY(VARCHAR), "vals_x": ARRAY(INT) }),
    extractions: [
        { outputName: "keys",   chain: [MapKeys],                              dataType: ARRAY(VARCHAR) },
        { outputName: "vals_x", chain: [MapValues, ArrayElements, StructField("x")],  dataType: ARRAY(INT) }
    ]
}
```

```
-- col: MAP(BIGINT, ROW(a: VARCHAR, b: DOUBLE, c: INT))
-- Query: SELECT cardinality(col) AS sz, map_values(col).a AS vals_a,
--        map_values(col).b AS vals_b FROM t
--
-- Three outputs: size and two value subfields.
-- Each subfield gets its own NamedExtraction with a StructField chain.

HiveColumnHandle {
    name: "col",
    hiveType: MAP(BIGINT, ROW(a: VARCHAR, b: DOUBLE, c: INT)),
    dataType: ROW({ "sz": BIGINT, "vals_a": ARRAY(VARCHAR), "vals_b": ARRAY(DOUBLE) }),
    extractions: [
        { outputName: "sz",     chain: [Size],                                     dataType: BIGINT },
        { outputName: "vals_a", chain: [MapValues, ArrayElements, StructField("a")],  dataType: ARRAY(VARCHAR) },
        { outputName: "vals_b", chain: [MapValues, ArrayElements, StructField("b")],  dataType: ARRAY(DOUBLE) }
    ]
}
```

## Future Extensions

### `ArraySlice` — sequence truncating for ML

ML models often consume variable-length sequences (user activity histories,
embedding arrays, feature maps) but have a maximum sequence length.  Reading
10K-element arrays when the model only uses the last 128 is wasteful.

`ArraySlice(offset, length)` would be a **type-preserving** step that truncates
arrays at the reader level.  A negative `length` selects from the end of the
array: `ArraySlice(0, -128)` means "take the last 128 elements."

| Step                   | Required input | Output      |
|------------------------|----------------|-------------|
| `ArraySlice(off, len)` | `ARRAY(T)`     | `ARRAY(T)`  |

Since the output type is the same as the input, `ArraySlice` does NOT require
`ArrayElements` after it (unlike `MapKeys`/`MapValues` which change the type).

Derivation rule:

```
derive(ARRAY(T), [ArraySlice, ...rest]) = derive(ARRAY(T), rest)
```

Composed examples:

```
-- Feature map with array values, filter to specific keys, truncate each to 128
col: MAP(VARCHAR, ARRAY(FLOAT))
Chain: [MapKeyFilter(["feat1", "feat2"]), MapValues, ArrayElements, ArraySlice(0, 128)]
Output: ARRAY(ARRAY(FLOAT))
```

```
-- User history, take first 128 events, extract only item_id
col: ARRAY(ROW(item_id BIGINT, timestamp BIGINT))
Chain: [ArraySlice(0, 128), ArrayElements, StructField("item_id")]
Output: ARRAY(BIGINT)
```

```
-- User history, take LAST 128 events (most recent)
col: ARRAY(ROW(item_id BIGINT, timestamp BIGINT))
Chain: [ArraySlice(0, -128)]
Output: ARRAY(ROW(item_id BIGINT, timestamp BIGINT))
```

```
-- Feature map with array values, filter to specific keys, take last 64 from each
col: MAP(VARCHAR, ARRAY(FLOAT))
Chain: [MapKeyFilter(["feat1", "feat2"]), MapValues, ArrayElements, ArraySlice(0, -64)]
Output: ARRAY(ARRAY(FLOAT))
```

This is a natural extension — it fits into the chain as a type-preserving step,
composable with extraction steps before and after it.  Velox already has
`ScanSpec::maxArrayElementsCount_` for the prefix case; `ArraySlice` generalizes
it to work within extraction chains.
