# Delimited Text Format

## INTEGER

These rules apply to TINYINT, SMALLINT, INTEGER, and BIGINT file columns.
Unlike REAL and DOUBLE, integer parsing does not trim whitespace.

### Accepted forms

Decimal integers have an optional `+` or `-` sign followed by one or more
decimal digits. Leading zeros are allowed. Examples include `0`, `42`, `+42`,
`-42`, and `+00042`.

A decimal point followed by zero or more digits is also allowed after the
integer digits. The fractional part is discarded, truncating toward zero:
`+12.9` reads as `12`, `-12.9` reads as `-12`, and `1.` reads as `1`.
The integer part must fit the file column's signed range: 8 bits for TINYINT,
16 bits for SMALLINT, 32 bits for INTEGER, and 64 bits for BIGINT.

### Inputs that read as NULL

- Empty fields and the configured null marker.
- Fields containing whitespace, including leading or trailing whitespace.
- A sign without digits, repeated signs, or a decimal point without preceding
  digits, such as `+`, `-`, `++1`, `+-1`, and `.5`.
- Exponents, type suffixes, hexadecimal forms, and other trailing characters,
  such as `1e2`, `1f`, `0x10`, `1x`, and `1.2x`.
- Multiple decimal points, such as `1.2.3`.
- Integer parts outside the file column's range. For example, `+128` reads as
  NULL for TINYINT, and `+9223372036854775808` reads as NULL for BIGINT.

## REAL and DOUBLE

The reader trims leading and trailing bytes whose unsigned value is at most
`0x20`. Parsing is locale-independent and uses `.` as the decimal separator.
The entire trimmed field must match an accepted form; malformed fields read
as NULL.

### Accepted forms

Decimal numbers have an optional `+` or `-` sign and at least one decimal digit.
They may contain a decimal point and an exponent (`e` or `E`, an optional sign,
and one or more digits). A single `f`, `F`, `d`, or `D` suffix is allowed
immediately after the number, with no intervening whitespace.

Examples include `1`, `1.25`, `+1.25D`, `-1.25f`, `1.`, `1.f`, `.5`, `-.5`,
`+.5f`, `1e2F`, and `1.25E-3d`.

The suffix does not select the parsing precision. REAL file columns are parsed
directly as `float`; DOUBLE file columns are parsed directly as `double`. A REAL
column requested as DOUBLE is parsed as `float` and then widened to `double`.
For example, `16777217f` reads as `16777216` in REAL and `16777217` in DOUBLE.

`NaN`, `+NaN`, and `-NaN` read as NaN.
NaN is case-sensitive and does not accept a type suffix or payload.

`inf` and `infinity` are accepted in any ASCII letter case, with an optional
`+` or `-` sign. Examples include `Infinity`, `+Infinity`, `-Inf`, `INF`, and
`inFINITY`. Infinity spellings do not accept a type suffix.

Rounding uses the file column's precision, with ties rounded to even. Overflow
produces signed infinity, underflow can produce zero, and representable
subnormal values are retained. For example, `1e100f` overflows REAL but remains
finite in DOUBLE.

### Inputs that read as NULL

- Empty fields, whitespace-only fields, and the configured null marker.
- Other NaN spellings, payloads, and suffixes, such as `nan`, `NAN`,
  `NaN(payload)`, and `NaNf`.
- Suffixed infinity spellings, such as `Infinityf` and `-Infd`.
- Missing mantissa or exponent digits, such as `.`, `.e2`, `1e`, `1e+`,
  and `1e+f`.
- A suffix without a number, repeated suffixes, or whitespace before a suffix,
  such as `f`, `1ff`, and `1.5 d`.
- Other trailing characters or embedded whitespace/control bytes, such as
  `1e5x`, `1.2.3`, and an embedded NUL.
- Hexadecimal forms, such as `0x10` and `0x1.8p+1`.

## Writer formatting

DOUBLE fields use the non-legacy VARCHAR cast formatter. Finite values retain
their significant digits rather than being rounded to six decimal places.
For example, `1.5` is written as `1.5`. Values with absolute magnitude below
`0.001` or at least `10000000` use scientific notation, except zero.
NaN and infinities are written as `NaN`, `Infinity`, and `-Infinity`.

The shared formatter honors the process-wide
`--experimental_enable_legacy_cast` flag when adding `.0` to whole numbers
in standard notation. With the default value of `false`, `1.0`, `0.0`, and
`-0.0` are written as `1.0`, `0.0`, and `-0.0`. When the flag is `true`, they
are written as `1`, `0`, and `-0`. The flag does not change scientific notation
selection in the text writer.

DOUBLE output is not guaranteed to match Hive's Java-based formatting byte
for byte.

DATE fields use the ISO date representation, such as `2024-01-01`, rather than
the number of days since the epoch.

DECIMAL fields use plain decimal notation with exactly the declared number of
fractional digits. For example, an unscaled value of `150` in DECIMAL(5,2) is
written as `1.50`. Both short and long decimals are supported, including scale
zero.

These conversions also apply to values nested in arrays, maps, and rows.
Null values use the configured null marker. REAL and TIMESTAMP formatting are
unchanged.
