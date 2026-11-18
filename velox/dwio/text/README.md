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

## DATE

Dates require a complete dashed year-month-day. The year may have a leading
`+` or `-`; month and day may each have one or two digits. Leading ASCII
whitespace is allowed. The date must be valid and fit in a signed 32-bit
number of days since 1970-01-01.

After the date, any suffix beginning with uppercase `T` or ASCII whitespace is
ignored. For example, `2024-01-01`, `+2024-1-1`, `2024-01-01T10:00`,
`2024-01-01Tnot-a-time`, and `2024-01-01 ignored` all read as the same date.
The suffix is not validated as a time.

Empty fields and the configured null marker (default `\N`) read as NULL.
Other malformed fields raise an error, rather than reading as NULL.
Examples include whitespace-only fields, incomplete dates such as `2024-01`,
invalid dates such as `2023-02-29`, other separators such as `2024/01/01`,
and suffixes beginning with another character such as `2024-01-01X10:00`
or `2024-01-01t10:00`.

## TIMESTAMP

A date without a time reads as midnight. A time may follow the date with a
space or uppercase `T` separator. Leading and trailing ASCII whitespace is
allowed. Existing date forms are preserved, including optional signs and
one- or two-digit month and day fields.

For the `T` form, the time must start immediately after `T` and contain an
hour and minute, with optional seconds and a fractional part after the seconds.
Hours, minutes, and seconds have one or two digits. Hours must be in [0, 23],
minutes in [0, 59], and seconds in [0, 59]. A fractional decimal point must be
followed by at least one digit. Examples include `2024-01-01T10:00`,
`2024-01-01T10:00:30`, and `2024-01-01T10:00:30.123456789`.

Fractions are truncated to microseconds, not rounded. For example, the last
example retains `.123456`. The existing whitespace-separated time grammar is
unchanged, including acceptance of `2024-01-01 00:00:60` and
`2024-01-01 00:00:001`. The corresponding `T` forms are rejected.

Parsed timestamps are interpreted in `America/Los_Angeles`, including its
seasonal offset, and converted to UTC. This preserves the reader's existing
timezone policy; it does not use a session timezone. Explicit timezone suffixes
and offsets, such as `Z` or `+01:00`, are not accepted.

Empty fields and the configured null marker (default `\N`) read as NULL.
Other malformed fields raise an error. Examples include whitespace-only
fields, `2024-01-01T`, `2024-01-01T 00:00:00`, `2024-01-01T24:00:00`,
`2024-01-01T00:00:60`, `2024-01-01T00:00:001`, and
`2024-01-01T00:00:00.`.

These are text-reader rules, not a full Hive grammar-conformance guarantee.
Ordinary SQL casts retain their existing parsing rules.
