# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import datetime
import decimal
import json
import socket
import sys
import threading
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from unittest import mock

import pyarrow as pa
from pyspark import cloudpickle
from pyspark.sql.types import Row
from velox.exec.python import worker

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

STRUCT = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
UNNAMED_PAIR = pa.struct([pa.field("", pa.int32()), pa.field("", pa.int32())])


@dataclass
class Spec:
    """A function for SETUP. Integer arguments are columns of BATCH; strings
    name earlier functions. An 'output' of None leaves the key out."""

    name: str
    function: Callable[..., Any]
    arguments: list[int | str]
    result_type: pa.DataType
    output: bool | None = None


EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def micros(moment: datetime.datetime) -> int:
    return (moment - EPOCH) // datetime.timedelta(microseconds=1)


def header(specs: list[Spec], session_time_zone: str = "UTC") -> dict[str, Any]:
    names = [spec.name for spec in specs]
    functions = []
    for spec in specs:
        function: dict[str, Any] = {
            "name": spec.name,
            "arguments": [
                {"column": argument}
                if isinstance(argument, int)
                else {"function": names.index(argument)}
                for argument in spec.arguments
            ],
        }
        if spec.output is not None:
            function["output"] = spec.output
        functions.append(function)
    return {
        "protocolVersion": worker.PROTOCOL_VERSION,
        "pythonVersion": PYTHON_VERSION,
        "sessionTimeZone": session_time_zone,
        "functions": functions,
    }


def setup_payload(
    specs: list[Spec], header_json: dict[str, Any] | None = None
) -> bytes:
    schema = pa.schema(
        [pa.field(f"f{i}", spec.result_type) for i, spec in enumerate(specs)]
    )
    return worker.pack_sections(
        [
            json.dumps(header_json or header(specs)).encode(),
            schema.serialize().to_pybytes(),
        ]
        + [cloudpickle.dumps((spec.function, None)) for spec in specs]
    )


class Connection:
    """A worker serving on a thread, and the executor's end of its socket."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.test = test
        self.client, self.server = socket.socketpair()
        self.client.settimeout(10)
        self.server.settimeout(10)
        self.stream = self.client.makefile("rwb")
        self.server_stream = self.server.makefile("rwb")
        self.status: int | None = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            self.status = worker.run(self.server_stream)
        finally:
            # Closing flushes; the test may have stopped reading.
            with contextlib.suppress(OSError):
                self.server_stream.close()
            self.server.close()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.stream.close()
        self.client.close()
        self.thread.join(timeout=10)
        self.test.assertFalse(self.thread.is_alive())

    def send(self, message_type: int, payload: bytes) -> None:
        worker.write_message(self.stream, message_type, payload)

    def receive(self) -> tuple[int, bytes]:
        return worker.read_message(self.stream)

    def start(self, specs: list[Spec], session_time_zone: str = "UTC") -> None:
        self.send(
            worker.SETUP,
            setup_payload(specs, header(specs, session_time_zone)),
        )
        message_type, payload = self.receive()
        self.test.assertEqual(message_type, worker.READY, payload)
        self.test.assertGreaterEqual(json.loads(payload)["unpickleSeconds"], 0)

    def send_batch(self, columns: list[Any]) -> None:
        batch = pa.record_batch(columns, names=[f"c{i}" for i in range(len(columns))])
        self.send(worker.BATCH, worker.write_batch(batch))

    def run_batch(self, columns: list[Any]) -> pa.RecordBatch:
        self.send_batch(columns)
        message_type, payload = self.receive()
        self.test.assertEqual(message_type, worker.RESULT, payload[:1000])
        return worker.read_batch(payload)

    def assert_error(self, message: str) -> None:
        message_type, payload = self.receive()
        self.test.assertEqual(message_type, worker.ERROR)
        error = json.loads(payload)
        self.test.assertEqual(error["message"], message)
        self.test.assertIn("Traceback", error["traceback"])
        self.thread.join(timeout=10)
        self.test.assertEqual(self.status, 1)

    def end(self) -> None:
        self.send(worker.END, b"")
        message_type, payload = self.receive()
        self.test.assertEqual(message_type, worker.END)
        stats = json.loads(payload)
        self.test.assertGreater(stats["cpuSeconds"], 0)
        self.test.assertGreater(stats["peakMemoryBytes"], 0)
        self.thread.join(timeout=10)
        self.test.assertEqual(self.status, 0)


def result_column(batch: pa.RecordBatch, output: int) -> pa.Array:
    """Returns the results of the output-th output function."""
    return batch.column(2 * output)


def error_column(batch: pa.RecordBatch, output: int) -> pa.Array:
    """Returns the errors of the output-th output function."""
    return batch.column(2 * output + 1)


def results(batch: pa.RecordBatch, output: int) -> tuple[list[Any], list[Any]]:
    return (
        result_column(batch, output).to_pylist(),
        error_column(batch, output).to_pylist(),
    )


def wall_seconds(batch: pa.RecordBatch) -> list[float]:
    metadata = batch.schema.metadata
    assert metadata is not None
    return json.loads(metadata[b"functionWallSeconds"])


def unnamed_pairs(num_rows: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [cast(pa.Array, pa.array([value] * num_rows, pa.int32())) for value in (1, 2)],
        fields=list(UNNAMED_PAIR),
    )


def invalid_utf8() -> pa.Array:
    """Returns a VARCHAR array of one value that is not valid UTF-8."""
    offsets = pa.py_buffer(b"\x00\x00\x00\x00\x02\x00\x00\x00")
    return pa.Array.from_buffers(
        pa.string(), 1, cast(Any, [None, offsets, pa.py_buffer(b"\xff\xfe")])
    )


def list_layouts() -> list[tuple[Any, list[str]]]:
    """Returns list arrays in layouts beyond the plain one, and how repr shows
    their values."""
    # A NULL list entry whose child range holds values, then sliced.
    offsets = cast(pa.Int32Array, pa.array([0, 2, 4, 5], pa.int32()))
    values = cast(pa.Array, pa.array([1, 2, 3, 4, 5], pa.int32()))
    mask = cast(Any, pa.array([False, True, False]))
    large_offsets = cast(Any, pa.array([0, 2, 4, 5], pa.int64()))
    return [
        (
            pa.ListArray.from_arrays(offsets, values, mask=mask).slice(1),
            ["None", "[5]"],
        ),
        (
            pa.LargeListArray.from_arrays(large_offsets, values, mask=mask).slice(1),
            ["None", "[5]"],
        ),
        (
            pa.array([[1], [2, 3], [4]], pa.list_(pa.int32())).slice(1),
            ["[2, 3]", "[4]"],
        ),
    ]


def map_and_struct_layouts() -> list[tuple[Any, list[str]]]:
    """Returns sliced map and struct arrays, and how repr shows their values."""
    map_type = pa.map_(pa.string(), pa.int32())
    maps = pa.array([[("a", 1)], [("b", 2)], [("c", 3)]], map_type)
    null_map_with_values = pa.MapArray.from_arrays(
        cast(Any, pa.array([0, 1, 2, 3], pa.int32())),
        cast(Any, pa.array(["a", "b", "c"])),
        cast(Any, pa.array([1, 2, 3], pa.int32())),
        mask=cast(Any, pa.array([False, True, False])),
    )
    lists_of_maps = pa.array(
        [[[("a", 1)]], [[("b", 2)]], [[("c", 3)]]], pa.list_(map_type)
    )
    structs = pa.array(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}], STRUCT
    )
    return [
        (maps.slice(1), ["{'b': 2}", "{'c': 3}"]),
        (null_map_with_values.slice(1), ["None", "{'c': 3}"]),
        (lists_of_maps.slice(1), ["[{'b': 2}]", "[{'c': 3}]"]),
        (structs.slice(1), ["Row(a=2, b='y')", "Row(a=3, b='z')"]),
    ]


def no_offsets(child_type: pa.DataType) -> pa.Array:
    """Returns an empty list array without offsets, which Arrow allows."""
    return pa.Array.from_buffers(
        pa.list_(child_type),
        0,
        cast(Any, [None, None]),
        children=[cast(pa.Array, pa.array([], child_type))],
    )


def nested_layouts() -> list[Any]:
    """Returns arrays of nested types with NULLs and empty values at each
    level, for the round trip."""
    timestamp = pa.timestamp("us")
    # 01:30 twice in America/Los_Angeles, on the night clocks fall back.
    fall_back = [
        micros(datetime.datetime(2025, 11, 2, hour, 30, tzinfo=datetime.timezone.utc))
        for hour in (8, 9)
    ]
    sliced_child = pa.ListArray.from_arrays(
        cast(Any, pa.array([0, 1, 3], pa.int32())),
        cast(Any, pa.array([1, 2, 3, 4], pa.int32()).slice(1)),
    ).slice(1)
    return [
        pa.array([1.5], pa.float64()),
        pa.array(
            [decimal.Decimal("123456789012345678901234567890.12")], pa.decimal128(38, 2)
        ),
        pa.array([7], pa.int16()),
        pa.array([7], pa.int32()),
        pa.array(fall_back, timestamp),
        pa.array([[{"a": 1, "b": "x"}, None], [], None], pa.list_(STRUCT)),
        pa.array(
            [{"m": [("k", [1, None]), ("e", [])]}, {"m": None}, None],
            pa.struct([pa.field("m", pa.map_(pa.string(), pa.list_(pa.int32())))]),
        ),
        pa.array([[("k", None)], [], None], pa.map_(pa.string(), pa.int32())),
        pa.array(
            [[(MOMENT_MICROS, [MOMENT_MICROS])]],
            pa.map_(timestamp, pa.list_(timestamp)),
        ),
        pa.array([{"a": None, "b": "x"}], STRUCT),
        # A NULL row whose fields hold values.
        pa.StructArray.from_arrays(
            [
                cast(pa.Array, pa.array([1, 2], pa.int32())),
                cast(pa.Array, pa.array(["x", "y"])),
            ],
            fields=list(STRUCT),
            mask=cast(Any, pa.array([False, True])),
        ),
        pa.array([[], [1]], pa.list_(pa.int32())),
        pa.array(
            [[({"a": 1, "b": [1, 2]}, 5)]],
            pa.map_(
                pa.struct(
                    [pa.field("a", pa.int32()), pa.field("b", pa.list_(pa.int32()))]
                ),
                pa.int32(),
            ),
        ),
        sliced_child,
        pa.StructArray.from_arrays(
            [
                cast(pa.Array, pa.array([MOMENT_MICROS], timestamp)),
                cast(pa.Array, pa.array([[1, 2]], pa.list_(pa.int32()))),
            ],
            fields=[pa.field("", timestamp), pa.field("", pa.list_(pa.int32()))],
        ),
        pa.ListArray.from_arrays(
            cast(Any, pa.array([0, 0], pa.int32())), no_offsets(pa.int32())
        ),
        pa.array([], pa.list_(pa.int32())),
        unnamed_pairs(2),
    ]


MOMENT = datetime.datetime(2025, 1, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)
MOMENT_MICROS = micros(MOMENT)
OUT_OF_RANGE = "argument is outside the years Python covers: 300000000000"

# Arguments and how repr shows the Python value a function receives.
ARGUMENT_CASES: list[tuple[Any, str]] = [
    (pa.array([True]), "True"),
    (pa.array([7], pa.int8()), "7"),
    (pa.array([7], pa.int64()), "7"),
    (pa.array([1.5], pa.float32()), "1.5"),
    (pa.array(["abc"]), "'abc'"),
    (pa.array(["abc"], pa.large_string()), "'abc'"),
    (pa.array([b"\x01\x02"]), "bytearray(b'\\x01\\x02')"),
    (pa.array([b"\x01"], pa.large_binary()), "bytearray(b'\\x01')"),
    (pa.array([datetime.date(2025, 1, 2)]), "datetime.date(2025, 1, 2)"),
    (pa.array([decimal.Decimal("1.25")], pa.decimal128(10, 2)), "Decimal('1.25')"),
    (pa.array([[1, None, 3]], pa.list_(pa.int32())), "[1, None, 3]"),
    (pa.array([[1]], pa.large_list(pa.int32())), "[1]"),
    (pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int32())), "{'k': 1}"),
    (pa.array([[(b"k", 1)]], pa.map_(pa.binary(), pa.int32())), "{b'k': 1}"),
    (
        pa.array([[([1, 2], 3)]], pa.map_(pa.list_(pa.int32()), pa.int32())),
        "{(1, 2): 3}",
    ),
    (pa.array([{"a": 1, "b": "x"}], STRUCT), "Row(a=1, b='x')"),
    (pa.array([None], pa.int32()), "None"),
    (pa.array([None], pa.list_(pa.int32())), "None"),
    (pa.array([None], pa.map_(pa.string(), pa.int32())), "None"),
    (pa.array([None], STRUCT), "None"),
]

# Timestamp arguments in Asia/Kathmandu, UTC+5:45 (UTC+5:30 until 1986), and
# how str shows them.
TIMESTAMP_ARGUMENT_CASES: list[tuple[Any, str]] = [
    (pa.array([MOMENT_MICROS // 1_000_000], pa.timestamp("s")), "2025-01-02 08:49:05"),
    (pa.array([MOMENT_MICROS // 1_000], pa.timestamp("ms")), "2025-01-02 08:49:05"),
    (pa.array([MOMENT_MICROS], pa.timestamp("us")), "2025-01-02 08:49:05.000006"),
    (
        pa.array([MOMENT_MICROS * 1_000], pa.timestamp("ns")),
        "2025-01-02 08:49:05.000006",
    ),
    (
        pa.array(
            [
                micros(
                    datetime.datetime(
                        1960, 5, 6, 7, 8, 9, 250_000, tzinfo=datetime.timezone.utc
                    )
                )
            ],
            pa.timestamp("us"),
        ),
        "1960-05-06 12:38:09.250000",
    ),
]

# Timestamps nested in Asia/Kathmandu, and how repr shows them.
NESTED_TIMESTAMP_ARGUMENT_CASES: list[tuple[Any, str]] = [
    (
        pa.array([[MOMENT_MICROS]], pa.list_(pa.timestamp("us"))),
        "[datetime.datetime(2025, 1, 2, 8, 49, 5, 6)]",
    ),
]

# Result types, values a function returns, and the stored values.
RESULT_CASES: list[tuple[pa.DataType, object, object]] = [
    (pa.bool_(), True, True),
    (pa.int8(), 127, 127),
    (pa.int64(), 2**40, 2**40),
    (pa.float64(), 1.5, 1.5),
    (pa.float64(), 7, 7.0),
    (pa.float32(), 1.5, 1.5),
    (pa.float32(), 2**24, float(2**24)),
    (pa.float32(), float("inf"), float("inf")),
    (pa.string(), "abc", "abc"),
    (pa.large_string(), "abc", "abc"),
    (pa.binary(), bytearray(b"ab"), b"ab"),
    (pa.binary(), b"ab", b"ab"),
    (pa.date32(), datetime.date(2025, 1, 2), datetime.date(2025, 1, 2)),
    (pa.decimal128(10, 2), decimal.Decimal("1.5"), decimal.Decimal("1.50")),
    (pa.decimal128(10, 2), decimal.Decimal("-0.00"), decimal.Decimal("0.00")),
    (
        pa.decimal128(38, 2),
        decimal.Decimal("123456789012345678901234567890.12"),
        decimal.Decimal("123456789012345678901234567890.12"),
    ),
    (pa.list_(pa.int32()), (1, None), [1, None]),
    (pa.large_list(pa.int32()), [1], [1]),
    (pa.map_(pa.string(), pa.int32()), {"k": 1}, [("k", 1)]),
    (STRUCT, Row(a=1, b="x"), {"a": 1, "b": "x"}),
    (STRUCT, Row(b="x", a=1), {"a": 1, "b": "x"}),
    (STRUCT, Row(a=1), {"a": 1, "b": None}),
    (STRUCT, (1, "x"), {"a": 1, "b": "x"}),
    (STRUCT, {"a": 1}, {"a": 1, "b": None}),
    (pa.int32(), None, None),
]

# Result types, values a function returns, and the row's error after
# '<function> result '.
RESULT_ERROR_CASES: list[tuple[pa.DataType, object, str]] = [
    # Python types pyarrow would accept.
    (pa.float64(), True, "must be DOUBLE: bool True"),
    (pa.int32(), 1.5, "must be INTEGER: float 1.5"),
    (pa.string(), 7, "must be VARCHAR: int 7"),
    (pa.string(), b"a", "must be VARCHAR: bytes b'a'"),
    (pa.binary(), "a", "must be VARBINARY: str 'a'"),
    (
        pa.date32(),
        datetime.datetime(2025, 1, 2),
        "must be DATE: datetime datetime.datetime(2025, 1, 2, 0, 0)",
    ),
    (pa.date32(), 5, "must be DATE: int 5"),
    (pa.timestamp("us"), 5, "must be TIMESTAMP: int 5"),
    (pa.decimal128(10, 2), 1, "must be DECIMAL(10, 2): int 1"),
    (pa.list_(pa.string()), "ab", "must be ARRAY<VARCHAR>: str 'ab'"),
    (pa.list_(pa.int64()), [1, "x"], "must be BIGINT: str 'x'"),
    (
        pa.map_(pa.string(), pa.int32()),
        [("k", 1)],
        "must be MAP<VARCHAR,INTEGER>: list [('k', 1)]",
    ),
    (pa.float32(), 1e40, "is out of range for REAL: 1e+40"),
    # Values pyarrow rejects.
    (
        pa.float64(),
        2**53 + 1,
        "does not fit DOUBLE: Integer value 9007199254740993 is outside of the "
        "range exactly representable by a IEEE 754 double precision value",
    ),
    (
        pa.float32(),
        2**24 + 1,
        "does not fit REAL: Integer value 16777217 is outside of the range "
        "exactly representable by a IEEE 754 single precision value",
    ),
    (
        pa.int8(),
        300,
        "does not fit TINYINT: Value 300 too large to fit in C integer type",
    ),
    (
        pa.float64(),
        2**53 + 2,
        "does not fit DOUBLE: Integer value 9007199254740994 is outside of the "
        "range exactly representable by a IEEE 754 double precision value",
    ),
    (
        pa.timestamp("ms"),
        datetime.datetime(2025, 1, 2, 0, 0, 0, 123456, tzinfo=datetime.timezone.utc),
        "is more precise than TIMESTAMP in ms: 2025-01-02 00:00:00.123456+00:00",
    ),
    (
        pa.list_(pa.int8()),
        [300],
        "does not fit ARRAY<TINYINT>: Value 300 too large to fit in C integer type",
    ),
    (
        pa.string(),
        "a\udc80",
        "does not fit VARCHAR: 'utf-8' codec can't encode character '\\udc80' in "
        "position 1: surrogates not allowed",
    ),
    (
        pa.timestamp("ns"),
        datetime.datetime(2300, 1, 1, tzinfo=datetime.timezone.utc),
        "does not fit TIMESTAMP: Python int too large to convert to C long",
    ),
    (
        pa.decimal128(10, 2),
        decimal.Decimal("1.255"),
        "does not fit DECIMAL(10, 2): Rescaling Decimal128 value would cause data loss",
    ),
    (
        pa.decimal128(10, 2),
        decimal.Decimal("123456789.00"),
        "does not fit DECIMAL(10, 2): Decimal type with precision 11 does not fit "
        "into precision inferred from first array element: 10",
    ),
    (
        pa.map_(pa.string(), pa.int32()),
        {None: 1},
        "does not fit MAP<VARCHAR,INTEGER>: Invalid Map: key field cannot contain "
        "null values",
    ),
    # Row shapes.
    (STRUCT, (1,), "must have 2 values for ROW<a:INTEGER,b:VARCHAR>: 1"),
    (
        pa.struct([pa.field("a", pa.int32())]),
        {"a": 1, "z": 2},
        "has fields that ROW<a:INTEGER> does not have: 'z'",
    ),
    (
        pa.struct([pa.field("a", pa.int32())]),
        {"a": 1, 1: 0, "b": 0},
        "has fields that ROW<a:INTEGER> does not have: 'b', 1",
    ),
    (
        UNNAMED_PAIR,
        {"": 1},
        "must give fields by position for ROW<:INTEGER,:INTEGER>, whose field names "
        "repeat",
    ),
    (
        STRUCT,
        Row("a", "a")(1, 2),
        "must not repeat field names other than ROW<a:INTEGER,b:VARCHAR>'s: ['a', 'a']",
    ),
]


def setup_error_cases() -> list[tuple[bytes, str]]:
    """Returns SETUP payloads the worker rejects, and its ERROR messages."""
    one = [Spec("f", repr, [0], pa.string())]
    good = header(one)

    def with_header(**changes: Any) -> bytes:
        return setup_payload(one, {**good, **changes})

    def with_function(**changes: Any) -> bytes:
        return with_header(functions=[{**good["functions"][0], **changes}])

    def with_result_type(result_type: pa.DataType) -> bytes:
        return setup_payload([Spec("f", repr, [0], result_type)])

    return [
        (
            with_header(pythonVersion="2.7"),
            f"Functions must be pickled with the worker's Python {PYTHON_VERSION}: 2.7",
        ),
        (with_header(protocolVersion=9), "Protocol version must be 1: 9"),
        (
            worker.pack_sections([json.dumps({"protocolVersion": 1}).encode(), b""]),
            "SETUP header must have key: pythonVersion",
        ),
        (
            worker.pack_sections([json.dumps(good).encode()]),
            "SETUP must have a header and a result schema",
        ),
        (with_header(sessionTimeZone="Z"), "Session time zone is unknown: Z"),
        (with_header(sessionTimeZone="+24:00"), "Session time zone is unknown: +24:00"),
        (with_header(sessionTimeZone="+05:99"), "Session time zone is unknown: +05:99"),
        (b"\x01\x00", "SETUP section length must not be truncated"),
        (setup_payload(one)[:-3], "SETUP section must not run past the end of SETUP"),
        (
            worker.pack_sections(
                [json.dumps(good).encode(), pa.schema([]).serialize().to_pybytes()]
            ),
            "SETUP must have as many result types and pickled functions as "
            "functions: 1, 0, 0",
        ),
        (with_header(functions=[{"name": "f"}]), "Function must have key: arguments"),
        (
            with_function(arguments=[{"column": True}]),
            "Argument must be a column or a function: {'column': True}",
        ),
        (
            with_function(arguments=[{"column": 0, "function": 0}]),
            "Argument must be a column or a function: {'column': 0, 'function': 0}",
        ),
        (
            with_function(arguments=[{"column": -1}]),
            "Argument column must not be negative: {'column': -1}",
        ),
        (
            with_function(arguments=[{"function": 0}]),
            "Function argument must refer to an earlier function: {'function': 0}",
        ),
        (with_function(output="yes"), "Function output must be true or false: f"),
        (with_function(output=False), "SETUP must have at least one output function"),
        (with_result_type(pa.time64("us")), "Type is not supported: time64[us]"),
        (with_result_type(pa.uint32()), "Type is not supported: uint32"),
        (with_result_type(pa.float16()), "Type is not supported: halffloat"),
        (
            with_result_type(
                pa.map_(pa.list_(pa.map_(pa.string(), pa.int32())), pa.int32())
            ),
            "Map keys must not contain maps: "
            "map<list<item: map<string, int32>>, int32>",
        ),
    ]


class WorkerTest(unittest.TestCase):
    def connect(self) -> Connection:
        connection = Connection(self)
        self.addCleanup(connection.close)
        return connection

    def assert_values(
        self,
        cases: list[tuple[Any, Any]],
        function: Callable[..., Any] = repr,
        session_time_zone: str = "UTC",
    ) -> None:
        """Checks that 'function' maps each case's array to its value."""
        connection = self.connect()
        connection.start(
            [Spec(f"f{i}", function, [i], pa.string()) for i in range(len(cases))],
            session_time_zone,
        )
        result = connection.run_batch([array for array, _ in cases])
        for i, (array, expected) in enumerate(cases):
            with self.subTest(type=str(array.type), index=i):
                self.assertEqual(results(result, i), (expected, [None] * len(expected)))
        connection.end()

    def assert_results(
        self, cases: list[tuple[pa.DataType, object, Any]], expected: Callable[..., Any]
    ) -> None:
        """Checks the row each case's returned value gives: 'expected' maps a
        case's index and expectation to its (results, errors)."""
        connection = self.connect()
        connection.start(
            [
                Spec(f"f{i}", lambda _, value=value: value, [0], result_type)
                for i, (result_type, value, _) in enumerate(cases)
            ]
        )
        result = connection.run_batch([pa.array([0])])
        for i, (result_type, value, expectation) in enumerate(cases):
            with self.subTest(type=str(result_type), value=repr(value), index=i):
                self.assertEqual(results(result, i), expected(i, expectation))
        connection.end()

    def test_arguments(self) -> None:
        self.assert_values([(array, [shown]) for array, shown in ARGUMENT_CASES])

    def test_round_trip(self) -> None:
        # A function returning its argument returns the same values, for
        # every type and layout, timestamps to the microsecond.
        arrays = [
            array
            for array, _ in ARGUMENT_CASES
            + TIMESTAMP_ARGUMENT_CASES
            + NESTED_TIMESTAMP_ARGUMENT_CASES
        ]
        arrays += [array for array, _ in list_layouts() + map_and_struct_layouts()]
        arrays += nested_layouts()
        for zone in ("Asia/Kathmandu", "America/Los_Angeles"):
            for i, array in enumerate(arrays):
                with self.subTest(type=str(array.type), index=i, zone=zone):
                    connection = self.connect()
                    connection.start(
                        [Spec("identity", lambda value: value, [0], array.type)],
                        session_time_zone=zone,
                    )
                    result = connection.run_batch([array])
                    self.assertEqual(error_column(result, 0).null_count, len(array))
                    self.assertTrue(result_column(result, 0).equals(array))
                    connection.end()

    def test_list_layouts(self) -> None:
        self.assert_values(list_layouts())

    def test_map_and_struct_layouts(self) -> None:
        self.assert_values(map_and_struct_layouts())
        # Struct fields with repeated names arrive by position.
        connection = self.connect()
        connection.start([Spec("unnamed", list, [0], pa.list_(pa.int32()))])
        result = connection.run_batch([unnamed_pairs(2)])
        self.assertEqual(results(result, 0), ([[1, 2], [1, 2]], [None, None]))
        connection.end()

    def test_timestamp_arguments(self) -> None:
        self.assert_values(
            [(array, [shown]) for array, shown in TIMESTAMP_ARGUMENT_CASES],
            function=str,
            session_time_zone="Asia/Kathmandu",
        )
        self.assert_values(
            [(array, [shown]) for array, shown in NESTED_TIMESTAMP_ARGUMENT_CASES],
            session_time_zone="Asia/Kathmandu",
        )
        moment = datetime.datetime(2025, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        self.assert_values(
            [(pa.array([micros(moment)], pa.timestamp("us")), ["2025-01-01 19:04:05"])],
            function=str,
            session_time_zone="-08:00",
        )

    def test_arguments_without_python_values(self) -> None:
        seconds = pa.timestamp("s")
        far_date = pa.array([3_000_000], pa.int32()).cast(pa.date32())
        connection = self.connect()
        connection.start(
            [
                Spec("timestamp", repr, [0], pa.string()),
                Spec("in_list", repr, [1], pa.string()),
                Spec("in_struct", repr, [2], pa.string()),
                Spec("in_map", repr, [3], pa.string()),
                Spec("date", repr, [4], pa.string()),
                Spec("varchar", repr, [5], pa.string()),
                Spec("varchar_key", repr, [6], pa.string()),
            ]
        )
        invalid_utf8_key = pa.MapArray.from_arrays(
            cast(Any, pa.array([0, 1], pa.int32())),
            cast(Any, invalid_utf8()),
            cast(Any, pa.array([1], pa.int32())),
        )
        result = connection.run_batch(
            [
                pa.array([300_000_000_000], seconds),
                pa.array([[300_000_000_000]], pa.list_(seconds)),
                pa.array([{"t": 300_000_000_000}], pa.struct([pa.field("t", seconds)])),
                pa.array([[("k", 300_000_000_000)]], pa.map_(pa.string(), seconds)),
                far_date,
                invalid_utf8(),
                invalid_utf8_key,
            ]
        )
        invalid = (
            "argument has no Python value: 'utf-8' codec can't decode byte 0xff in "
            "position 0: invalid start byte"
        )
        for i, message in enumerate(
            [f"timestamp {OUT_OF_RANGE}"]
            + [f"{name} {OUT_OF_RANGE}" for name in ("in_list", "in_struct", "in_map")]
            + [
                "date argument has no Python value: date value out of range",
                f"varchar {invalid}",
                f"varchar_key {invalid}",
            ]
        ):
            self.assertEqual(results(result, i), ([None], [message]))
        connection.end()

    def test_timestamp_results(self) -> None:
        timestamp = pa.timestamp("us")
        connection = self.connect()
        connection.start(
            [
                Spec("identity", lambda t: t, [0], timestamp),
                Spec(
                    "naive",
                    lambda t: datetime.datetime(2025, 1, 2, 8, 49, 5, 6),
                    [0],
                    timestamp,
                ),
                Spec("aware", lambda t: MOMENT, [0], timestamp),
                Spec(
                    "in_struct",
                    lambda t: {"t": t},
                    [0],
                    pa.struct([pa.field("t", timestamp)]),
                ),
                Spec("in_list", lambda t: [t], [0], pa.list_(timestamp)),
                Spec(
                    "in_map", lambda t: {"k": t}, [0], pa.map_(pa.string(), timestamp)
                ),
            ],
            session_time_zone="Asia/Kathmandu",
        )
        result = connection.run_batch([pa.array([MOMENT_MICROS], timestamp)])
        for i in range(3):
            self.assertEqual(
                result_column(result, i).cast(pa.int64()).to_pylist(), [MOMENT_MICROS]
            )
        self.assertEqual(
            result_column(result, 3)
            .cast(pa.struct([pa.field("t", pa.int64())]))
            .to_pylist(),
            [{"t": MOMENT_MICROS}],
        )
        self.assertEqual(
            result_column(result, 4).cast(pa.list_(pa.int64())).to_pylist(),
            [[MOMENT_MICROS]],
        )
        self.assertEqual(
            result_column(result, 5).cast(pa.map_(pa.string(), pa.int64())).to_pylist(),
            [[("k", MOMENT_MICROS)]],
        )
        self.assertEqual([results(result, i)[1] for i in range(6)], [[None]] * 6)
        connection.end()

    def test_results(self) -> None:
        self.assert_results(RESULT_CASES, lambda i, stored: ([stored], [None]))

    def test_result_errors(self) -> None:
        self.assert_results(
            RESULT_ERROR_CASES, lambda i, message: ([None], [f"f{i} result {message}"])
        )

    def test_nested_result_types(self) -> None:
        # The declared list field name survives, though pyarrow's type
        # equality ignores it.
        element_list = pa.list_(pa.field("element", pa.int32()))
        connection = self.connect()
        connection.start(
            [
                Spec("list", lambda _: [1], [0], element_list),
                Spec("identity", lambda row: row, [1], UNNAMED_PAIR),
                Spec("pair", lambda row: (3, 4), [1], UNNAMED_PAIR),
            ]
        )
        result = connection.run_batch([pa.array([0]), unnamed_pairs(1)])
        self.assertEqual(str(result_column(result, 0).type), str(element_list))
        # pyarrow reads repeated names back into one dict entry; compare the
        # fields themselves.
        for i, expected in [(1, [1, 2]), (2, [3, 4])]:
            structs = cast(pa.StructArray, result_column(result, i))
            fields = [structs.field(j).to_pylist()[0] for j in range(2)]
            self.assertEqual(fields, expected)
        self.assertEqual(
            [error_column(result, i).to_pylist() for i in range(3)], [[None]] * 3
        )
        connection.end()

    def test_without_pyspark(self) -> None:
        with mock.patch.object(worker, "_row_class", return_value=None):
            connection = self.connect()
            connection.start(
                [
                    Spec("pair", lambda x: {"a": x, "b": "x"}, [0], STRUCT),
                    Spec("row", repr, [1], pa.string()),
                ]
            )
            # An unread ROW column needs no pyspark.
            result = connection.run_batch(
                [
                    pa.array([1], pa.int32()),
                    pa.array([None], pa.int32()),
                    pa.array([{"a": 1, "b": "x"}], STRUCT),
                ]
            )
            self.assertEqual(results(result, 0), ([{"a": 1, "b": "x"}], [None]))
            self.assertEqual(results(result, 1), (["None"], [None]))
            message = (
                "ProtocolError: ROW arguments require pyspark, which is not installed"
            )
            # A ROW column a function reads needs pyspark, even with no ROW
            # values in the batch.
            for column in [
                pa.array([], STRUCT),
                pa.array([[], None], pa.list_(STRUCT)),
            ]:
                with self.subTest(type=str(column.type)):
                    connection = self.connect()
                    connection.start([Spec("row", repr, [0], pa.string())])
                    connection.send_batch([column])
                    connection.assert_error(message)
            # A ROW result another function reads needs pyspark too, which
            # SETUP checks.
            connection = self.connect()
            specs = [
                Spec("pair", lambda x: (x, "x"), [0], STRUCT, output=False),
                Spec("row", repr, ["pair"], pa.string()),
            ]
            connection.send(worker.SETUP, setup_payload(specs))
            connection.assert_error(message)

    def test_results_fitting_only_row_by_row(self) -> None:
        build = worker._array

        def fail_together(values: list[Any], arrow_type: pa.DataType) -> pa.Array:
            if len(values) > 1:
                raise ValueError("too large together")
            return build(values, arrow_type)

        with mock.patch.object(worker, "_array", side_effect=fail_together):
            connection = self.connect()
            connection.start([Spec("f", lambda x: x, [0], pa.int64())])
            result = connection.run_batch([pa.array([1, 2])])
            message = "f result does not fit BIGINT: too large together"
            self.assertEqual(results(result, 0), ([None, None], [message, message]))
            connection.end()

    def test_exceptions(self) -> None:
        def leave(x: int) -> int:
            sys.exit(0)

        def surrogate(x: int) -> int:
            raise ValueError("a\udc80")

        connection = self.connect()
        connection.start(
            [
                Spec("inverse", lambda x: 1 / x, [0], pa.float64()),
                Spec("double_inverse", lambda x: 2 / x, [0], pa.float64()),
                Spec("leave", leave, [0], pa.int64()),
                Spec("surrogate", surrogate, [0], pa.int64()),
            ]
        )
        result = connection.run_batch([pa.array([1, 0, 4, 0])])
        for index, name, values in [
            (0, "inverse", [1.0, None, 0.25, None]),
            (1, "double_inverse", [2.0, None, 0.5, None]),
        ]:
            message = f"{name} raised ZeroDivisionError: division by zero"
            function_results, errors = results(result, index)
            self.assertEqual(function_results, values)
            self.assertEqual([errors[0], errors[2], errors[3]], [None, None, message])
            # Each function's first failed row carries the traceback.
            self.assertTrue(errors[1].startswith(f"{message}\nTraceback"), errors[1])
        function_results, errors = results(result, 2)
        self.assertEqual(function_results, [None] * 4)
        self.assertTrue(errors[0].startswith("leave raised SystemExit: 0\nTraceback"))
        self.assertEqual(errors[1:], ["leave raised SystemExit: 0"] * 3)
        # A message UTF-8 cannot encode arrives escaped.
        errors = results(result, 3)[1]
        self.assertTrue(errors[0].startswith("surrogate raised ValueError: a\\udc80\n"))
        self.assertEqual(errors[1:], ["surrogate raised ValueError: a\\udc80"] * 3)
        connection.end()

    def test_nested_functions(self) -> None:
        connection = self.connect()
        # Evaluates, with c0 the batch's column 0:
        #   describe(inverse_or_seven(c0))
        #   add_one(inverse_or_seven(c0))
        #   upper(describe_add_one(add_one(inverse_or_seven(c0))))
        connection.start(
            [
                # Returns the int 7 for a DOUBLE result, stored as 7.0; later
                # functions read the stored value.
                Spec(
                    "inverse_or_seven",
                    lambda x: 1 / x if x != 2 else 7,
                    [0],
                    pa.float64(),
                    output=False,
                ),
                Spec("describe", repr, ["inverse_or_seven"], pa.string()),
                Spec("add_one", lambda v: v + 1, ["inverse_or_seven"], pa.float64()),
                Spec("describe_add_one", repr, ["add_one"], pa.string(), output=False),
                Spec("upper", lambda s: s.upper(), ["describe_add_one"], pa.string()),
            ]
        )
        result = connection.run_batch([pa.array([1, 0, 2])])
        # Only the three output functions return columns.
        self.assertEqual(result.num_columns, 6)
        inner_error = (
            "inverse_or_seven raised ZeroDivisionError: division by zero\nTraceback"
        )
        for index, values, prefix in [
            (
                0,
                ["1.0", None, "7.0"],
                f"describe input inverse_or_seven failed: {inner_error}",
            ),
            (
                1,
                [2.0, None, 8.0],
                f"add_one input inverse_or_seven failed: {inner_error}",
            ),
            (
                2,
                ["2.0", None, "8.0"],
                "upper input describe_add_one failed: describe_add_one input add_one "
                f"failed: add_one input inverse_or_seven failed: {inner_error}",
            ),
        ]:
            function_results, errors = results(result, index)
            self.assertEqual(function_results, values)
            self.assertEqual([errors[0], errors[2]], [None, None])
            # The inner function's whole error, traceback included, carries over.
            self.assertTrue(errors[1].startswith(prefix), errors[1])
        connection.end()

    def test_shared_columns(self) -> None:
        connection = self.connect()
        connection.start(
            [
                Spec("difference", lambda a, b: a - b, [0, 1], pa.int64()),
                Spec("reverse_difference", lambda b, a: b - a, [1, 0], pa.int64()),
                Spec("square", lambda b, b_again: b * b_again, [1, 1], pa.int64()),
            ]
        )
        result = connection.run_batch([pa.array([10]), pa.array([3])])
        self.assertEqual(
            [results(result, i) for i in range(3)],
            [([7], [None]), ([-7], [None]), ([9], [None])],
        )
        connection.end()

    def test_wall_seconds(self) -> None:
        def slow(x: int) -> int:
            time.sleep(0.05)
            return x

        connection = self.connect()
        connection.start(
            [
                Spec("constant", lambda: 1, [], pa.int64()),
                Spec("slow", slow, [0], pa.int64(), output=False),
                Spec("identity", lambda x: x, ["slow"], pa.int64()),
            ]
        )
        result = connection.run_batch([pa.array([7])])
        self.assertEqual(
            [results(result, i) for i in range(2)], [([1], [None]), ([7], [None])]
        )
        # Every function's time, output or not, in order.
        seconds = wall_seconds(result)
        self.assertEqual(len(seconds), 3)
        self.assertGreaterEqual(seconds[1], 0.05)
        connection.end()

    def test_several_batches(self) -> None:
        connection = self.connect()
        connection.start(
            [
                Spec("add", lambda a, b: a + b, [0, 1], pa.int64()),
                Spec("upper", lambda s: s.upper(), [2], pa.string()),
                Spec("length", len, [3], pa.int64()),
            ]
        )
        # The executor may send the next BATCH before reading a RESULT.
        for offset in range(3):
            connection.send_batch(
                [
                    pa.array([offset, 1]),
                    pa.array([10, 20]),
                    pa.array(["a", "b"]),
                    pa.array([[1], []], pa.list_(pa.int32())),
                ]
            )
        for offset in range(3):
            message_type, payload = connection.receive()
            self.assertEqual(message_type, worker.RESULT)
            result = worker.read_batch(payload)
            self.assertEqual(results(result, 0), ([offset + 10, 21], [None, None]))
            self.assertEqual(results(result, 1), (["A", "B"], [None, None]))
            self.assertEqual(results(result, 2), ([1, 0], [None, None]))
        connection.end()

    def test_empty_batch(self) -> None:
        connection = self.connect()
        connection.start([Spec("length", len, [0], pa.int64())])
        # An empty list column may have no offsets at all.
        result = connection.run_batch([no_offsets(pa.int32())])
        self.assertEqual(results(result, 0), ([], []))
        connection.end()

    def test_batch_without_columns(self) -> None:
        connection = self.connect()
        connection.start([Spec("constant", lambda: 1, [], pa.int64())])
        batch = pa.RecordBatch.from_struct_array(
            cast(pa.StructArray, pa.array([{}] * 3, pa.struct([])))
        )
        connection.send(worker.BATCH, worker.write_batch(batch))
        message_type, payload = connection.receive()
        self.assertEqual(message_type, worker.RESULT)
        self.assertEqual(results(worker.read_batch(payload), 0), ([1] * 3, [None] * 3))
        connection.end()

    def test_results_failing_on_some_rows(self) -> None:
        def tiny(x: int) -> int:
            if x == 2:
                raise ValueError("two")
            return x

        connection = self.connect()
        connection.start(
            [
                Spec("tiny", tiny, [0], pa.int8()),
                Spec("describe", repr, ["tiny"], pa.string()),
            ]
        )
        result = connection.run_batch([pa.array([1, 2, 300, 4])])
        raised = "tiny raised ValueError: two"
        too_large = (
            "tiny result does not fit TINYINT: Value 300 too large to fit in C "
            "integer type"
        )
        values, errors = results(result, 0)
        self.assertEqual(values, [1, None, None, 4])
        self.assertTrue(errors[1].startswith(f"{raised}\nTraceback"), errors[1])
        self.assertEqual([errors[0], errors[2], errors[3]], [None, too_large, None])
        values, errors = results(result, 1)
        self.assertEqual(values, ["1", None, None, "4"])
        self.assertTrue(errors[1].startswith(f"describe input tiny failed: {raised}"))
        self.assertEqual(errors[2], f"describe input tiny failed: {too_large}")
        connection.end()

    def test_map_keys_equal_in_python(self) -> None:
        def map_of(keys: list[Any], key_type: pa.DataType) -> Any:
            return pa.array(
                [[(key, i) for i, key in enumerate(keys)]],
                pa.map_(key_type, pa.int32()),
            )

        fall_back = [
            micros(
                datetime.datetime(2025, 11, 2, hour, 30, tzinfo=datetime.timezone.utc)
            )
            for hour in (8, 9)
        ]
        connection = self.connect()
        connection.start(
            [
                Spec("nanos", repr, [0], pa.string()),
                Spec("fall_back", repr, [1], pa.string()),
                # Keys without Python values fail for that reason, not as equal.
                Spec("far_dates", repr, [2], pa.string()),
            ],
            session_time_zone="America/Los_Angeles",
        )
        result = connection.run_batch(
            [
                map_of([1, 2], pa.timestamp("ns")),
                map_of(fall_back, pa.timestamp("us")),
                map_of([3_000_000, 3_000_001], pa.date32()),
            ]
        )
        prefix = "argument map has keys equal in Python"
        self.assertEqual(
            [results(result, i) for i in range(3)],
            [
                ([None], [f"nanos {prefix}: datetime.datetime(1969, 12, 31, 16, 0)"]),
                (
                    [None],
                    [
                        f"fall_back {prefix}: "
                        "datetime.datetime(2025, 11, 2, 1, 30, fold=1)"
                    ],
                ),
                (
                    [None],
                    ["far_dates argument has no Python value: date value out of range"],
                ),
            ],
        )
        connection.end()

    def test_map_arguments_without_rows_do_not_load_row(self) -> None:
        with mock.patch.object(worker, "_row_class") as row_class:
            connection = self.connect()
            connection.start(
                [Spec("keys", lambda m: sorted(m), [0], pa.list_(pa.string()))]
            )
            result = connection.run_batch(
                [pa.array([[("b", 1), ("a", 2)]], pa.map_(pa.string(), pa.int32()))]
            )
            self.assertEqual(results(result, 0), ([["a", "b"]], [None]))
            connection.end()
        row_class.assert_not_called()

    def test_setup_errors(self) -> None:
        for payload, message in setup_error_cases():
            with self.subTest(message=message):
                connection = self.connect()
                connection.send(worker.SETUP, payload)
                connection.assert_error(f"ProtocolError: {message}")

    def test_pickled_function_errors(self) -> None:
        schema = pa.schema([pa.field("f0", pa.string())]).serialize().to_pybytes()
        good = json.dumps(header([Spec("f", repr, [0], pa.string())])).encode()
        for code, message in [
            (cloudpickle.dumps(7), "Pickled function must be callable: f"),
            (cloudpickle.dumps(()), "Pickled function must be callable: f"),
            (
                b"\x80\x04cno_such_module\nfunction\n.",
                "Pickled function failed to unpickle: f: ModuleNotFoundError: "
                "No module named 'no_such_module'",
            ),
            # Calls sys.exit(0).
            (
                b"\x80\x04csys\nexit\nK\x00\x85R.",
                "Pickled function failed to unpickle: f: SystemExit: 0",
            ),
        ]:
            with self.subTest(message=message):
                connection = self.connect()
                connection.send(
                    worker.SETUP, worker.pack_sections([good, schema, code])
                )
                connection.assert_error(f"ProtocolError: {message}")

    def test_batch_errors(self) -> None:
        connection = self.connect()
        connection.start([Spec("f", repr, [1], pa.string())])
        connection.send_batch([pa.array([1])])
        connection.assert_error(
            "ProtocolError: Argument column must be one of BATCH's 1: 1"
        )

        connection = self.connect()
        connection.start([Spec("f", repr, [0], pa.string())])
        connection.send_batch([pa.array([1], pa.time32("s"))])
        connection.assert_error("ProtocolError: Type is not supported: time32[s]")

        connection = self.connect()
        connection.start([Spec("f", repr, [0], pa.string())])
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, pa.schema([pa.field("c0", pa.int64())])):
            pass
        connection.send(worker.BATCH, sink.getvalue().to_pybytes())
        connection.assert_error("ProtocolError: BATCH must hold one record batch: 0")

        connection = self.connect()
        connection.start([Spec("f", repr, [0], pa.string())])
        batch = pa.record_batch([pa.array([1])], names=["c0"])
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
            writer.write_batch(batch)
        connection.send(worker.BATCH, sink.getvalue().to_pybytes())
        connection.assert_error("ProtocolError: BATCH must hold one record batch: 2")

        connection = self.connect()
        connection.start([Spec("f", repr, [0], pa.string())])
        connection.send_batch([pa.array([1], pa.timestamp("us", "UTC"))])
        connection.assert_error(
            "ProtocolError: Type is not supported: timestamp[us, tz=UTC]"
        )

    def test_message_errors(self) -> None:
        connection = self.connect()
        connection.send(worker.BATCH, b"")
        connection.assert_error("ProtocolError: First message must be SETUP: 3")

        connection = self.connect()
        connection.start([Spec("f", repr, [0], pa.string())])
        connection.send(worker.READY, b"")
        connection.assert_error("ProtocolError: Message must be BATCH or END: 2")

        for data, message in [
            (b"\x01\x10", "Connection must not close before END"),
            (
                worker._FRAME.pack(worker.SETUP, 16) + b"abc",
                "Connection must not close in the middle of a message",
            ),
        ]:
            with self.subTest(message=message):
                connection = self.connect()
                connection.stream.write(data)
                connection.stream.flush()
                connection.client.shutdown(socket.SHUT_WR)
                connection.assert_error(f"ProtocolError: {message}")

    def test_main(self) -> None:
        executor, worker_end = socket.socketpair()
        executor.settimeout(10)
        status: list[int] = []
        thread = threading.Thread(
            target=lambda: status.append(
                worker.main(["--fd", str(worker_end.detach())])
            ),
            daemon=True,
        )
        thread.start()
        stream = executor.makefile("rwb")
        worker.write_message(
            stream, worker.SETUP, setup_payload([Spec("f", repr, [0], pa.string())])
        )
        self.assertEqual(worker.read_message(stream)[0], worker.READY)
        worker.write_message(stream, worker.END, b"")
        self.assertEqual(worker.read_message(stream)[0], worker.END)
        thread.join(timeout=10)
        self.assertEqual(status, [0])
        stream.close()
        executor.close()
