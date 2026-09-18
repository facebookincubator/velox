# NimbleDSL

NimbleDSL is an interactive SQL-like interface for inspecting Nimble files. It provides a user-friendly way to query row data, view schema information, and inspect file metadata such as layout, encodings, and indexes.

This document gives a brief overview of the commands available in NimbleDSL. For more detailed information.


## Usage

```bash
./nimble_dsl -- <file>
```

`<file>` is the path to a Nimble file.

Type `HELP` at the prompt for a summary of all commands. Type `QUIT`, `EXIT`, or press Ctrl-D to exit.

## Commands

All commands are **case-insensitive**. Trailing semicolons and commas are optional.

### SELECT

Reads and displays row data from the file. (The default limit is 20 rows)

```
SELECT * [LIMIT n] [OFFSET n] [STRIPE s]
SELECT col1, col2 [LIMIT n] [OFFSET n] [STRIPE s]
```


- `*` selects all columns. Specific columns can be listed by name (space or comma separated).
- `LIMIT n` — return at most `n` rows.
- `OFFSET n` — skip the first `n` rows before returning results.
- `STRIPE s` — restrict reading to stripe `s` (0-indexed). The offset is relative to the start of that stripe.
- A `FROM` clause is accepted and silently ignored for SQL familiarity.

Examples:

```
nimble> SELECT *
nimble> SELECT name, age LIMIT 5
nimble> SELECT * LIMIT 10 OFFSET 100
nimble> SELECT * LIMIT 50 STRIPE 0
```

If an invalid column name is provided, an error message is printed suggesting the `DESCRIBE` command.
If an invalid stripe ID is provided, an error is printed showing the valid stripe count.

### DESCRIBE

```
DESCRIBE
```

Shows a table of top-level column names, their types, and the corresponding Nimble stream offsets.

Example output:

```
Column                    Type                 Stream
------------------------------------------------------------
user_id                   BIGINT               0
name                      VARCHAR              2
scores                    ARRAY<DOUBLE>        4
```

### SHOW SCHEMA

```
SHOW SCHEMA
```

Shows the full Nimble schema tree including nested types (arrays, maps, rows, flat maps) with stream offsets and type kinds. This is useful for understanding the complete structure of complex nested schemas.

### SHOW INFO

```
SHOW INFO
```

Shows file-level metadata:

- Nimble format version (major.minor)
- File size
- Checksum value and type (e.g. `XXH3_64`)
- Stripe count
- Row count
- User-defined metadata key-value pairs (e.g. `build.revision`, `hostname`)

### SHOW STATS

```
SHOW STATS
```

Shows per-column statistics: value count, null count, min, max, logical size, and physical size.

* Requires the file to have been written with `enableVectorizedStats = true`.
* If no statistics are available, a message is printed instead.

### SHOW STRIPES

```
SHOW STRIPES
```

Shows stripe-level information: stripe ID, byte offset, byte size, and row count for each stripe in the file.

### SHOW STREAMS

```
SHOW STREAMS [STRIPE s]
```

Shows stream-level information: stream ID, byte offset, byte length, item count, and stream label. Without `STRIPE`, shows streams across all stripes.
With `STRIPE s`, filters to a single stripe.

### SHOW ENCODING

```
SHOW ENCODING [STRIPE s]
```

Shows the encoding tree for each stream: encoding types, data types, and compression used. Without `STRIPE`, shows encodings across all stripes.
With `STRIPE s`, filters to a single stripe.

### SHOW INDEX

```
SHOW INDEX
```

Shows index information if the file has an index configured: index columns with sort orders (ASC/DESC NULLS LAST), number of stripes and index groups, per-group compression and offset details, and per-stripe key stream regions.
If no index is configured, a message is printed instead.

### HELP

```
HELP
```

Prints a summary of all available commands.

### QUIT / EXIT

```
QUIT
EXIT
```

Exits the REPL. Ctrl-D (EOF) also exits.
