=====
Types
=====

Velox supports scalar types and complex types. These types cover most of the
Presto and Spark data types.

Scalar Types
~~~~~~~~~~~~
Scalar types in Velox are logical and SQL-compatible.
Each scalar type is implemented using a C++ type. The table
below shows the supported scalar types and their corresponding C++ type.

======================  ===========================    ==================
Velox Type              C++ Type                       Bytes per Value
======================  ===========================    ==================
BOOLEAN                 bool                            0.125 (i.e. 1 bit)
TINYINT                 int8_t                          1
SMALLINT                int16_t                         2
INTEGER                 int32_t	                        4
BIGINT                  int64_t                         8
DATE                    struct Date                     8
REAL                    float                           4
DOUBLE                  double                          8
DECIMAL                 int64_t / int128_t              8 / 16
TIMESTAMP               struct Timestamp               16
INTERVAL DAY TO SECOND  int64_t                        8
VARCHAR                 struct StringView              16
VARBINARY               struct StringView              16
======================  ===========================    ==================

DECIMAL type is a logical type and carries additional `precision`,
and `scale` information. `Precision` is the number of
digits in a number. `Scale` is the number of digits to the right of the decimal
point in a number. For example, the number `123.45` has a precision of `5` and a
scale of `2`. Decimal types are backed by `int64_t` and `int128_t` C++ types which
store the unscaled value.
`int64_t` is used upto 18 precision, and has a range of
[:math:`-10^{18} + 1, +10^{18} - 1`]. `int128_t` is used starting from 19 precision
upto 38 precision, with a range of [:math:`-10^{38} + 1, +10^{38} - 1`].
For example, the unscaled value of decimal `123.45` is `12345`.
All the three values, precision, scale, unscaled value are required to represent a
decimal value.
Some systems use two `int64_t` values instead of a single `int128_t` value.
Velox chose `int128_t` since most compilers now support this type and
it simplifies the implementation.

Presto Types
~~~~~~~~~~~~
Velox also supports certain types specific to Presto.

========================  =========================    ==================
Presto Type               C++ Type                     Bytes per Value
========================  =========================    ==================
HYPERLOGLOG               struct StringView            16
JSON                      struct StringView            16
TIMESTAMP WITH TIME ZONE  struct<int64_t, int16_t>     20
========================  =========================    ==================

Complex Types
~~~~~~~~~~~~~
Velox supports the ARRAY, MAP, and ROW complex types.
