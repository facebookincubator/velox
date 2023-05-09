=====
Types
=====

Velox supports a fixed set of physical types, an extensible set of logical types,
complex types, and few types specific to Presto.


Physical Types
~~~~~~~~~~~~~~
Each physical type is implemented using a C++ type. The table
below shows the supported physical types, their corresponding C++ type,
and bytes required per value.

======================  ===========================    ==================
Velox Physical Type     C++ Type                       Bytes per Value
======================  ===========================    ==================
BOOLEAN                 bool                            0.125 (i.e. 1 bit)
TINYINT                 int8_t                          1
SMALLINT                int16_t                         2
INTEGER                 int32_t	                        4
BIGINT                  int64_t                         8
HUGEINT                 int128_t                       16
DATE                    struct Date                     8
REAL                    float                           4
DOUBLE                  double                          8
TIMESTAMP               struct Timestamp               16
VARCHAR                 struct StringView              16
VARBINARY               struct StringView              16
======================  ===========================    ==================

All physical types have a one-to-one mapping with their C++ types.

Logical Types
~~~~~~~~~~~~~
Velox logical types are backed by a physical type.
The table below shows the supported logical types, and
their corresponding physical type.

======================  ===========================
Velox Logical Type      Physical Type
======================  ===========================
DECIMAL                 BIGINT / HUGEINT
INTERVAL DAY TO SECOND  BIGINT
======================  ===========================

DECIMAL type carries additional `precision`,
and `scale` information. `Precision` is the number of
digits in a number. `Scale` is the number of digits to the right of the decimal
point in a number. For example, the number `123.45` has a precision of `5` and a
scale of `2`. Decimal types are backed by `BIGINT` and `HUGEINT` physical types,
which store the unscaled value. For example, the unscaled value of decimal
`123.45` is `12345`. `BIGINT` is used upto 18 precision, and has a range of
[:math:`-10^{18} + 1, +10^{18} - 1`]. `HUGEINT` is used starting from 19 precision
upto 38 precision, with a range of [:math:`-10^{38} + 1, +10^{38} - 1`].

All the three values, precision, scale, unscaled value are required to represent a
decimal value.

Complex Types
~~~~~~~~~~~~~
Velox supports the ARRAY, MAP, and ROW complex types.

Presto Types
~~~~~~~~~~~~
Velox also supports certain types specific to Presto that are backed by a
physical type.
The table below shows the supported Presto types, and
their corresponding physical type.

========================  =====================
Presto Type               Physical Type
========================  =====================
HYPERLOGLOG               VARCHAR
JSON                      VARCHAR
TIMESTAMP WITH TIME ZONE  ROW<BIGINT, SMALLINT>
========================  =====================

