# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate tiny INT96 fixtures without a writer that suppresses INT96 stats.

Run with Python 3 in this directory. Each uncompressed file has two required
columns 'ts' and 'id' and two row groups, with one PLAIN page per column.
Both groups have timestamps 0s, 1s, 2s; their ids are 0, 1, 2 and 3, 4, 5.
The footer stores genuine Julian-day/nanos-of-day 12-byte bounds in either
modern min_value/max_value or deprecated min/max. Encoding follows the
Parquet Thrift schema and Thrift compact protocol; no runtime test dependency.
"""

from pathlib import Path
import struct


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(value):
    return varint(value * 2)  # All metadata integers here are nonnegative.


def binary(value):
    return varint(len(value)) + value


def sequence(kind, values):
    assert len(values) < 15
    return bytes([(len(values) << 4) | kind]) + b"".join(values)


def record(fields):
    result = bytearray()
    previous = 0
    for number, kind, value in fields:
        assert 0 < number - previous < 16
        result.append(((number - previous) << 4) | kind)
        result.extend(value)
        previous = number
    return bytes(result) + b"\0"


# Compact protocol types: I32=5, I64=6, BINARY=8, LIST=9, STRUCT=12.
values = [struct.pack("<qI", seconds * 1_000_000_000, 2440588) for seconds in range(3)]
page_data = b"".join(values)
data_header = record(
    [(1, 5, integer(3)), (2, 5, integer(0)), (3, 5, integer(3)), (4, 5, integer(3))]
)
page_header = record(
    [
        (1, 5, integer(0)),
        (2, 5, integer(len(page_data))),
        (3, 5, integer(len(page_data))),
        (5, 12, data_header),
    ]
)
page = page_header + page_data
id_data = struct.pack("<qqq", 0, 1, 2)
id_page = (
    record(
        [
            (1, 5, integer(0)),
            (2, 5, integer(len(id_data))),
            (3, 5, integer(len(id_data))),
            (5, 12, data_header),
        ]
    )
    + id_data
)
schema = sequence(
    12,
    [
        record([(4, 8, binary(b"schema")), (5, 5, integer(2))]),
        record([(1, 5, integer(3)), (3, 5, integer(0)), (4, 8, binary(b"ts"))]),
        record([(1, 5, integer(2)), (3, 5, integer(0)), (4, 8, binary(b"id"))]),
    ],
)
for name, max_field, min_field in [("modern", 5, 6), ("deprecated", 1, 2)]:
    stats = record(
        [(max_field, 8, binary(values[-1])), (min_field, 8, binary(values[0]))]
    )
    pages = bytearray()
    groups = []
    for first_id in (0, 3):
        offset = 4 + len(pages)
        group_id_page = id_page[: -len(id_data)] + struct.pack(
            "<qqq", first_id, first_id + 1, first_id + 2
        )
        column_metadata = record(
            [
                (1, 5, integer(3)),
                (2, 9, sequence(5, [integer(0)])),
                (3, 9, sequence(8, [binary(b"ts")])),
                (4, 5, integer(0)),
                (5, 6, integer(3)),
                (6, 6, integer(len(page))),
                (7, 6, integer(len(page))),
                (9, 6, integer(offset)),
                (12, 12, stats),
            ]
        )
        column = record([(2, 6, integer(offset)), (3, 12, column_metadata)])
        id_metadata = record(
            [
                (1, 5, integer(2)),
                (2, 9, sequence(5, [integer(0)])),
                (3, 9, sequence(8, [binary(b"id")])),
                (4, 5, integer(0)),
                (5, 6, integer(3)),
                (6, 6, integer(len(id_page))),
                (7, 6, integer(len(id_page))),
                (9, 6, integer(offset + len(page))),
                (
                    12,
                    12,
                    record(
                        [
                            (1, 8, binary(struct.pack("<q", first_id + 2))),
                            (2, 8, binary(struct.pack("<q", first_id))),
                            (5, 8, binary(struct.pack("<q", first_id + 2))),
                            (6, 8, binary(struct.pack("<q", first_id))),
                        ]
                    ),
                ),
            ]
        )
        id_column = record([(2, 6, integer(offset + len(page))), (3, 12, id_metadata)])
        group = record(
            [
                (1, 9, sequence(12, [column, id_column])),
                (2, 6, integer(len(page) + len(id_page))),
                (3, 6, integer(3)),
            ]
        )
        groups.append(group)
        pages.extend(page + group_id_page)
    footer = record(
        [
            (1, 5, integer(1)),
            (2, 9, schema),
            (3, 6, integer(6)),
            (4, 9, sequence(12, groups)),
        ]
    )
    Path(__file__).with_name(f"int96_statistics_{name}.parquet").write_bytes(
        b"PAR1" + pages + footer + struct.pack("<I", len(footer)) + b"PAR1"
    )
