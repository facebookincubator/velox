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

DATE parsing follows Hive 4.0.0's `LazyDate` and `Date.valueOf`. The reader
rejects fields shorter than eight bytes before parsing. It trims leading and
trailing bytes whose unsigned value is at most `0x20`, then parses a complete
dashed year-month-day prefix. The year has 1-10 digits with an optional `-`,
but no `+`; negative zero years are rejected. Month and day each have one or
two digits. The date must be valid and fit in a signed 32-bit number of days
since 1970-01-01.

Any remaining suffix is ignored, even when it begins with a letter, digit,
or punctuation. For example, `2024-01-01T10:00`, `2024-01-01X10:00`,
`2024-01-01t10:00`, `2024-01-0110:00`, and `2024-01-01 ignored` all read as
`2024-01-01`. The suffix is not validated as a time.

Empty fields, the configured null marker (default `\N`), and malformed fields
read as NULL. Malformed examples include whitespace-only fields, incomplete
dates such as `2024-01`, invalid dates such as `2023-02-29`, other separators
such as `2024/01/01`, and a positive sign such as `+2024-01-01`.

## TIMESTAMP

TIMESTAMP parsing follows Hive 4.0.0's default `Timestamp.valueOf` rules.
Leading and trailing bytes whose unsigned value is at most `0x20` are trimmed.
Both a date and time are required. Two forms are accepted:

- Space-separated: `[-]year-month-day hour:minute:second[.fraction]`.
  The year has 1-10 digits and no leading `+`; the other fields each have
  one or two digits. There must be exactly one space between the date and
  time. Seconds are required. A fractional decimal point must be followed by
  1-9 digits. For example, `2024-1-2 3:4:5.123456789` is valid.
- ISO local date-time: `yyyy-MM-ddTHH:mm[:ss[.fraction]]`.
  Lowercase `t` is also accepted. Month, day, hour, minute, and second fields
  must each have two digits. Years have 4-10 digits, with a `+` for
  positive years wider than four digits and a `-` for negative years.
  Seconds are optional; a fraction is allowed only after seconds and may
  contain 0-9 digits. For example, `2024-01-02T03:04`,
  `2024-01-02t03:04:05.123456789`, and `2024-01-02T03:04:05.` are valid.

Both forms reject negative zero years and invalid calendar dates. Hours must
be in [0, 23]; minutes and seconds must be in [0, 59]. Fractional seconds retain
nanosecond precision. Values must fit within Velox's timestamp range.

Parsed timestamps are interpreted in `America/Los_Angeles`, including its
seasonal offset, and converted to UTC. This preserves the reader's existing
timezone policy; it does not use a session timezone. Explicit timezone suffixes
and offsets, such as `Z` or `+01:00`, are not accepted.

Empty fields, the configured null marker (default `\N`), and malformed fields
read as NULL. Malformed examples include a date without a time (`2024-01-01`),
missing seconds after a space (`2024-01-01 10:00`), one-digit ISO fields
(`2024-1-2T10:00`), leap seconds (`2024-01-01 00:00:60`), three-digit seconds
(`2024-01-01 00:00:001`), an empty space-form fraction
(`2024-01-01 00:00:00.`), and fractions longer than nine digits.

These rules apply only to the text reader; ordinary SQL casts retain their
existing parsing rules. Custom Hive `timestamp.formats` are not supported.
