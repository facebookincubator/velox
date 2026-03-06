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
| Mechanism      | `requiredSubfields` on column handle | `extraction` chain on column handle (new)          |
| Composability  | N/A                                  | Result can be further pruned via requiredSubfields |

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

Column extraction solves this by creating **separate column handles** for each
reference.  The same source column appears twice in the scan's `assignments`
with different extraction chains and output types:

```
assignments: {
    "a": { name: "col", extraction: [Size],    dataType: BIGINT          }
    "b": { name: "col", extraction: [],        dataType: MAP(VARCHAR, BIGINT),
           requiredSubfields: ["col[\"foo\"]"]                            }
}
```

The reader can now optimize each handle independently:

- For `a`: read only the lengths stream, skip keys and values entirely.
- For `b`: read keys and values, filter to key `"foo"`.

This separation is the core reason column extraction is a new mechanism rather
than an extension of subfield pruning.  Subfield pruning merges all references
to a column into one handle, which forces the reader to satisfy the union of
all requirements.  Column extraction allows each reference to specify its own
read strategy.

## Protocol: Extraction Chain

The extraction is expressed as an ordered chain of steps.  Each step operates on
one nesting level of the source type.  The chain is a new field on the column
handle, separate from `requiredSubfields`.

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

### Composition with subfield pruning

Column extraction and subfield pruning are orthogonal.  They compose through a
clear ordering:

1. **Extraction** is applied first.  It transforms the source type into a
   (possibly different) output type.
2. **Subfield pruning** (`requiredSubfields`) is applied second.  It operates on
   the **post-extraction type**, not the file type.

The extraction chain determines the output type.  `requiredSubfields` then
prunes within that output type using the same rules as today (struct members
→ null-fill, map keys → drop, array indices → truncate).

**Rule:** `requiredSubfields` paths are relative to the post-extraction type.
If extraction produces `ARRAY(ROW(x: INT, y: INT))`, then `requiredSubfields`
paths start from `ARRAY(ROW(x, y))`, not from the original file column type.

Examples:

```
-- map_values(col).x, map_values(col).y
-- where col: MAP(K, ROW(x: INT, y: INT, z: INT))
extraction:          [MapValues]
post-extraction type: ARRAY(ROW(x: INT, y: INT, z: INT))
requiredSubfields:   ["[*].x", "[*].y"]
-- "[*]" refers to array elements, ".x"/".y" prune the ROW to fields x and y.
-- Field z is null-filled.  StructField cannot replace this because multiple
-- fields are needed — StructField extracts a single field.
```

```
-- transform(map_values(col), m -> element_at(m, 'foo'))
-- where col: MAP(K1, MAP(VARCHAR, ROW(x: INT, y: INT, z: INT)))
-- and only x and y are used from the value
extraction:          [MapValues, ArrayElements, MapKeyFilter(["foo"])]
post-extraction type: ARRAY(MAP(VARCHAR, ROW(x: INT, y: INT, z: INT)))
requiredSubfields:   ["[*][*].x", "[*][*].y"]
-- MapKeyFilter already filters to key "foo", so requiredSubfields uses [*]
-- for the map (all remaining entries), not ["foo"] again.
-- ".x"/".y" prune the ROW.  Field z is null-filled.
```

```
-- map_keys(col) where col: MAP(ROW(a: INT, b: INT, c: INT), V)
-- and only a and b are used from each key
extraction:          [MapKeys]
post-extraction type: ARRAY(ROW(a: INT, b: INT, c: INT))
requiredSubfields:   ["[*].a", "[*].b"]
-- Extracts keys (which are ROWs), then prunes to fields a and b.
-- Field c is null-filled.
```

**When extraction is empty**, `requiredSubfields` operates on the file type
directly — this is the existing behavior, unchanged.

**When both are present**, the column handle carries:
- `hiveType`: file type (for the reader to know what to read)
- `extraction`: chain to transform the type (applied by the reader)
- `dataType`: post-extraction type (for validation)
- `requiredSubfields`: pruning paths relative to `dataType` (applied by the
  reader after extraction)

### Overlap between `MapKeyFilter` and map key pruning

`MapKeyFilter` and subfield pruning's map key subscripts (`["key1"]`, `["key2"]`)
both filter a map to specific keys.  They overlap in functionality but differ in
where they sit:

- **Subfield pruning** (`requiredSubfields: ["col[\"foo\"]"]`): filters keys on
  the source column, output stays `MAP(K, V)`.  No extraction chain involved.
- **`MapKeyFilter`**: filters keys as a step in the extraction chain, output
  stays `MAP(K, V)`.  Can compose with other extraction steps.

When key filtering is the **only** operation on a column (no other extraction
steps), either mechanism works.  Prefer subfield pruning in this case — it uses
the existing infrastructure with no new protocol fields.

When key filtering is **part of an extraction chain** (e.g.,
`[MapValues, ArrayElements, MapKeyFilter(["foo"])]`), use `MapKeyFilter`.
Subfield pruning cannot express key filtering at an intermediate nesting level
within the chain.

When an extraction chain already exists and `requiredSubfields` contains a
single map key subscript that could be expressed as `MapKeyFilter`, prefer
extending the chain with `MapKeyFilter` over using `requiredSubfields`.  This
keeps all filtering logic in the extraction chain, making the reader's job
simpler.

### Overlap between `StructField` and struct subfield pruning

`StructField` and subfield pruning's nested field paths (`.field1`, `.field2`)
both navigate into struct fields.  They overlap when accessing a single struct
child:

- **Subfield pruning** (`requiredSubfields: ["col.x"]`): prunes the struct to
  only field `x`, output stays `ROW(...)` with other fields null-filled.
- **`StructField("x")`**: extracts field `x` as a step in the extraction chain,
  output becomes the field's type directly (e.g., `INT`).

When struct field access is the **only** operation on a column, prefer subfield
pruning — it keeps the `ROW` type and uses the existing infrastructure.

When struct navigation is **part of an extraction chain** (e.g.,
`[StructField("a"), StructField("b"), MapKeys]`), use `StructField`.  It
navigates to the target type for subsequent extraction steps.  Subfield pruning
cannot compose with extraction — it only prunes within the post-extraction
type.

When an extraction chain already exists and `requiredSubfields` contains a
single struct field path that could be expressed as `StructField`, prefer
extending the chain with `StructField` over using `requiredSubfields`.  This
keeps all navigation logic in the extraction chain.

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

Extraction chain already exists (`MapValues`), and only a single struct field
is accessed.  Extend the chain with `StructField` instead of using
`requiredSubfields`.

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
subfield `x`.  Since there is already an extraction chain and only a single
struct field is accessed, extend the chain with `StructField` instead of using
`requiredSubfields`.

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

class HiveColumnHandle : public connector::ColumnHandle {
 public:
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<ExtractionPathElement> extraction = {},
      ...);

  /// Chain of extraction steps.  Empty means no extraction (current behavior).
  /// When non-empty:
  ///   - hiveType is the source column type in the file.
  ///   - dataType is the output type after extraction.
  ///   - requiredSubfields operates on the post-extraction type.
  const std::vector<ExtractionPathElement>& extraction() const {
    return extraction_;
  }

 private:
  ...
  std::vector<ExtractionPathElement> extraction_;
};
```

### `HiveColumnHandle` (Java / Presto coordinator)

```java
public class HiveColumnHandle extends BaseHiveColumnHandle {
    ...
    // Existing
    private final List<Subfield> requiredSubfields;

    // New: extraction chain
    private final List<ExtractionPathElement> extraction;

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

The extraction chain is serialized as a JSON array in the plan fragment,
alongside `requiredSubfields`:

```json
{
  "name": "col",
  "hiveType": "map(varchar, array(map(integer, double)))",
  "typeSignature": "array(array(array(integer)))",
  "requiredSubfields": [],
  "extraction": [
    {"step": "MAP_VALUES"},
    {"step": "ARRAY_ELEMENTS"},
    {"step": "ARRAY_ELEMENTS"},
    {"step": "MAP_KEYS"}
  ]
}
```

With key filter and struct field extraction:

```json
{
  "name": "col",
  "hiveType": "map(varchar, row(x integer, y integer))",
  "typeSignature": "array(integer)",
  "requiredSubfields": [],
  "extraction": [
    {"step": "MAP_KEY_FILTER", "stringFilterKeys": ["foo", "bar"]},
    {"step": "MAP_VALUES"},
    {"step": "ARRAY_ELEMENTS"},
    {"step": "STRUCT_FIELD", "fieldName": "x"}
  ]
}
```

### Contract

- `extraction` is empty → current behavior, no type transformation.
- `extraction` is non-empty → worker applies the chain, producing `dataType`.
- `requiredSubfields` operates on the **post-extraction type**, not the file
  type.
- Multiple column handles can reference the same source column with different
  extractions (e.g., one for keys, one for values, one for size).
- The worker validates `hiveType` + `extraction` chain → `dataType` and
  rejects mismatches.

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
Chain: [ArraySlice(0, 128)]
Output: ARRAY(ROW(item_id BIGINT, timestamp BIGINT))
requiredSubfields: ["[*].item_id"]
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
