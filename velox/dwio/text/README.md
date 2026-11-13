# Delimited Text Format

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

`NaN`, `+NaN`, and `-NaN` read as the same canonical quiet NaN, with no sign bit.
NaN is case-sensitive and does not accept a type suffix or payload.

`inf` and `infinity` are accepted in any ASCII letter case, with an optional
`+` or `-` sign. Examples include `Infinity`, `+Infinity`, `-Inf`, `INF`, and
`inFINITY`. Infinity spellings do not accept a type suffix.

Rounding uses the file column's precision, with ties rounded to even. Overflow
produces signed infinity, underflow can produce signed zero, and representable
subnormal values are retained. For example, `1e100f` overflows REAL but remains
finite in DOUBLE. Negative zero retains its sign.

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
