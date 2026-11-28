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

"""Runs Python functions on Arrow batches for Velox's PythonCall operator.

The operator's executor starts this script in the query's Python environment
and talks to it over an inherited socket:

    <interpreter> worker.py --fd <n>

README.md in this directory defines the protocol and how Velox values map to
Python values.
"""

import argparse
import datetime
import decimal
import importlib
import io
import json
import math
import pickle
import re
import socket
import struct
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa

PROTOCOL_VERSION = 1

SETUP = 1
READY = 2
BATCH = 3
RESULT = 4
ERROR = 5
END = 6

_FRAME = struct.Struct("<BQ")
_SECTION = struct.Struct("<I")

_UNITS_PER_SECOND = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}

# pyspark's Row once loaded, or False when pyspark is not installed.
_row: Any = None


def _row_class() -> Any:
    """Returns pyspark's Row, which ROW arguments arrive as, or None when
    pyspark is not installed. Imported on first use rather than at the top:
    pyspark may be missing, and importing it slows every worker's start."""
    global _row
    if _row is None:
        try:
            _row = importlib.import_module("pyspark.sql.types").Row
        except ImportError:
            _row = False
    return _row or None


# Checks that a result is a value of a type and turns it into a value pyarrow
# builds an array of that type from. Raises ResultError otherwise.
ResultConverter = Callable[[Any], Any]


class ResultError(Exception):
    """A result that is not a value of the declared type. The message completes
    '<function> result ...'."""


class ProtocolError(Exception):
    """A malformed or unexpected message."""


# Compared by identity, so that two in one map never merge as equal keys.
@dataclass(frozen=True, eq=False)
class Unconvertible:
    """An argument value a function cannot receive: a DATE or TIMESTAMP
    outside the years datetime covers, a VARCHAR that is not valid UTF-8, a
    MAP with keys equal in Python, or the result of an earlier function that
    failed. Calling a function on it fails the row with 'reason'."""

    reason: str


# Messages.


def read_message(stream: io.BufferedIOBase) -> tuple[int, bytes]:
    """Reads one message: its type and payload."""
    header = stream.read(_FRAME.size)
    if len(header) < _FRAME.size:
        raise ProtocolError("Connection must not close before END")
    message_type, length = _FRAME.unpack(header)
    payload = stream.read(length)
    if len(payload) < length:
        raise ProtocolError("Connection must not close in the middle of a message")
    return message_type, payload


def write_message(stream: io.BufferedIOBase, message_type: int, payload: bytes) -> None:
    stream.write(_FRAME.pack(message_type, len(payload)))
    stream.write(payload)
    stream.flush()


def pack_sections(sections: Sequence[bytes]) -> bytes:
    """Returns 'sections' framed as SETUP frames them."""
    return b"".join(_SECTION.pack(len(section)) + section for section in sections)


def split_sections(payload: bytes) -> list[bytes]:
    """Returns the sections of a SETUP payload."""
    sections = []
    offset = 0
    while offset < len(payload):
        if offset + _SECTION.size > len(payload):
            raise ProtocolError("SETUP section length must not be truncated")
        (length,) = _SECTION.unpack_from(payload, offset)
        offset += _SECTION.size
        if offset + length > len(payload):
            raise ProtocolError("SETUP section must not run past the end of SETUP")
        sections.append(payload[offset : offset + length])
        offset += length
    return sections


def read_batch(payload: bytes) -> pa.RecordBatch:
    batches = list(pa.ipc.open_stream(pa.py_buffer(payload)))
    if len(batches) != 1:
        raise ProtocolError(f"BATCH must hold one record batch: {len(batches)}")
    return batches[0]


def write_batch(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


# Types.


def _is_list(arrow_type: pa.DataType) -> bool:
    return pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)


def _is_string(arrow_type: pa.DataType) -> bool:
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def _is_binary(arrow_type: pa.DataType) -> bool:
    return pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type)


def _is_signed_integer(arrow_type: pa.DataType) -> bool:
    return arrow_type in (pa.int8(), pa.int16(), pa.int32(), pa.int64())


def _is_floating(arrow_type: pa.DataType) -> bool:
    return arrow_type in (pa.float32(), pa.float64())


def _type_name(arrow_type: pa.DataType) -> str:
    """Returns the Velox name of 'arrow_type', for error messages."""
    names = {
        pa.bool_(): "BOOLEAN",
        pa.int8(): "TINYINT",
        pa.int16(): "SMALLINT",
        pa.int32(): "INTEGER",
        pa.int64(): "BIGINT",
        pa.float32(): "REAL",
        pa.float64(): "DOUBLE",
        pa.date32(): "DATE",
    }
    if arrow_type in names:
        return names[arrow_type]
    if _is_string(arrow_type):
        return "VARCHAR"
    if _is_binary(arrow_type):
        return "VARBINARY"
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    if isinstance(arrow_type, pa.Decimal128Type):
        decimal_type = cast(pa.Decimal128Type, arrow_type)
        return f"DECIMAL({decimal_type.precision}, {decimal_type.scale})"
    if _is_list(arrow_type):
        return f"ARRAY<{_type_name(cast(pa.ListType, arrow_type).value_type)}>"
    if pa.types.is_map(arrow_type):
        map_type = cast(pa.MapType, arrow_type)
        return f"MAP<{_type_name(map_type.key_type)},{_type_name(map_type.item_type)}>"
    if pa.types.is_struct(arrow_type):
        fields = ",".join(
            f"{struct_field.name}:{_type_name(struct_field.type)}"
            for struct_field in cast(pa.StructType, arrow_type)
        )
        return f"ROW<{fields}>"
    return str(arrow_type)


def check_supported(arrow_type: pa.DataType) -> None:
    """Raises ProtocolError for a type the value mapping does not cover."""
    if _is_list(arrow_type):
        check_supported(cast(pa.ListType, arrow_type).value_type)
    elif pa.types.is_map(arrow_type):
        key_type = cast(pa.MapType, arrow_type).key_type
        # Python map keys must be hashable, and a dict is not.
        if _contains_map(key_type):
            raise ProtocolError(f"Map keys must not contain maps: {arrow_type}")
        check_supported(key_type)
        check_supported(cast(pa.MapType, arrow_type).item_type)
    elif pa.types.is_struct(arrow_type):
        for struct_field in cast(pa.StructType, arrow_type):
            check_supported(struct_field.type)
    elif not (
        pa.types.is_boolean(arrow_type)
        or _is_signed_integer(arrow_type)
        or _is_floating(arrow_type)
        or _is_string(arrow_type)
        or _is_binary(arrow_type)
        or pa.types.is_date32(arrow_type)
        or (
            pa.types.is_timestamp(arrow_type)
            and cast(pa.TimestampType, arrow_type).tz is None
        )
        or isinstance(arrow_type, pa.Decimal128Type)
    ):
        raise ProtocolError(f"Type is not supported: {arrow_type}")


def _contains_map(arrow_type: pa.DataType) -> bool:
    if pa.types.is_map(arrow_type):
        return True
    if _is_list(arrow_type):
        return _contains_map(cast(pa.ListType, arrow_type).value_type)
    if pa.types.is_struct(arrow_type):
        return any(
            _contains_map(struct_field.type)
            for struct_field in cast(pa.StructType, arrow_type)
        )
    return False


def _contains_struct(arrow_type: pa.DataType) -> bool:
    if pa.types.is_struct(arrow_type):
        return True
    if _is_list(arrow_type):
        return _contains_struct(cast(pa.ListType, arrow_type).value_type)
    if pa.types.is_map(arrow_type):
        map_type = cast(pa.MapType, arrow_type)
        return _contains_struct(map_type.key_type) or _contains_struct(
            map_type.item_type
        )
    return False


def _check_readable(arrow_type: pa.DataType) -> None:
    """Raises ProtocolError if functions cannot read values of 'arrow_type':
    ROW values arrive as pyspark Rows. Checked by type, so that it does not
    depend on whether a batch holds any ROW value."""
    if _contains_struct(arrow_type) and _row_class() is None:
        raise ProtocolError("ROW arguments require pyspark, which is not installed")


def _timestamps_as_int64(arrow_type: pa.DataType) -> pa.DataType:
    """Returns 'arrow_type' with its timestamps replaced by int64, which holds
    a timestamp's value in its own unit."""
    if pa.types.is_timestamp(arrow_type):
        return pa.int64()
    if pa.types.is_list(arrow_type):
        return pa.list_(_timestamps_as_int64(cast(pa.ListType, arrow_type).value_type))
    if pa.types.is_large_list(arrow_type):
        return pa.large_list(
            _timestamps_as_int64(cast(pa.LargeListType, arrow_type).value_type)
        )
    if pa.types.is_map(arrow_type):
        map_type = cast(pa.MapType, arrow_type)
        return pa.map_(
            _timestamps_as_int64(map_type.key_type),
            _timestamps_as_int64(map_type.item_type),
        )
    if pa.types.is_struct(arrow_type):
        return pa.struct(
            [
                pa.field(
                    struct_field.name,
                    _timestamps_as_int64(struct_field.type),
                    struct_field.nullable,
                )
                for struct_field in cast(pa.StructType, arrow_type)
            ]
        )
    return arrow_type


def parse_time_zone(name: str) -> datetime.tzinfo:
    """Returns the time zone named 'name': an IANA name such as
    America/Los_Angeles, or a fixed offset such as +05:45."""
    offset = re.fullmatch(r"([+-])(\d{2}):(\d{2})", name)
    if offset is None:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            raise ProtocolError(f"Session time zone is unknown: {name}") from None
    hours = int(offset.group(2))
    minutes = int(offset.group(3))
    if hours >= 24 or minutes >= 60:
        raise ProtocolError(f"Session time zone is unknown: {name}")
    sign = 1 if offset.group(1) == "+" else -1
    minutes += hours * 60
    return datetime.timezone(datetime.timedelta(minutes=sign * minutes))


# Arguments: Arrow arrays to the Python values a function receives.


def _column_values(array: pa.Array, zone: datetime.tzinfo) -> list[Any]:
    """Returns the values of 'array' as a function receives them. A value with
    no Python value, at any depth, makes its row's value an Unconvertible."""
    values = _values(array, zone)
    arrow_type = array.type
    if not (
        _is_list(arrow_type)
        or pa.types.is_map(arrow_type)
        or pa.types.is_struct(arrow_type)
    ):
        return values
    return [_first_unconvertible(value) or value for value in values]


def _first_unconvertible(value: Any) -> Unconvertible | None:
    """Returns the first Unconvertible in 'value', at any depth, or None."""
    if isinstance(value, Unconvertible):
        return value
    if isinstance(value, dict):
        children = [*value.keys(), *value.values()]
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return None
    for child in children:
        found = _first_unconvertible(child)
        if found is not None:
            return found
    return None


def _values(array: pa.Array, zone: datetime.tzinfo) -> list[Any]:
    """Returns the Python values of 'array', with an Unconvertible for each
    value that has none, at its depth."""
    arrow_type = array.type
    if len(array) == 0:
        # pyarrow's flatten() crashes the process on an empty list array
        # without offsets.
        return []
    if _is_list(arrow_type):
        lists = cast(pa.ListArray, array)
        return _split(_values(lists.flatten(), zone), lists.value_lengths(), list)
    if pa.types.is_map(arrow_type):
        map_type = cast(pa.MapType, arrow_type)
        entries = cast(
            pa.ListArray,
            array.cast(
                pa.list_(
                    pa.struct(
                        [
                            pa.field("key", map_type.key_type, nullable=False),
                            pa.field("value", map_type.item_type),
                        ]
                    )
                )
            ),
        )
        pairs = cast(pa.StructArray, entries.flatten())
        keys = _values(pairs.field(0), zone)
        items = _values(pairs.field(1), zone)
        return _split(
            list(zip(map(_hashable, keys), items)), entries.value_lengths(), _map_value
        )
    if pa.types.is_struct(arrow_type):
        return _struct_values(cast(pa.StructArray, array), zone)
    if pa.types.is_timestamp(arrow_type):
        units = _UNITS_PER_SECOND[cast(pa.TimestampType, arrow_type).unit]
        return [
            None if value is None else _session_time(value, units, zone)
            for value in array.cast(pa.int64()).to_pylist()
        ]
    values = _leaf_values(array)
    if _is_binary(arrow_type):
        return [
            bytearray(value) if isinstance(value, bytes) else value for value in values
        ]
    return values


def _map_value(pairs: list[tuple[Any, Any]]) -> dict[Any, Any] | Unconvertible:
    """Returns the dict of a map's key-value pairs. Keys that differ in Velox
    can be equal in Python, such as timestamps within a microsecond; dropping
    one would lose an entry."""
    value = dict(pairs)
    if len(value) == len(pairs):
        return value
    seen = set()
    for key, _ in pairs:
        if key in seen:
            return Unconvertible(f"argument map has keys equal in Python: {key!r}")
        seen.add(key)
    raise AssertionError("A shorter dict must have a repeated key")


def _leaf_values(array: pa.Array) -> list[Any]:
    """Returns pyarrow's Python values of 'array', with an Unconvertible for
    each value pyarrow cannot convert, such as a VARCHAR that is not valid
    UTF-8 or a DATE outside the years Python covers."""
    try:
        return array.to_pylist()
    except (ValueError, OverflowError):
        return [_leaf_value(array, row) for row in range(len(array))]


def _leaf_value(array: pa.Array, row: int) -> Any:
    try:
        return array[row].as_py()
    except (ValueError, OverflowError) as e:
        return Unconvertible(f"argument has no Python value: {e}")


def _session_time(
    value: int, units: int, zone: datetime.tzinfo
) -> datetime.datetime | Unconvertible:
    seconds, remainder = divmod(value, units)
    try:
        moment = datetime.datetime.fromtimestamp(seconds, zone)
    except (OverflowError, OSError, ValueError):
        return Unconvertible(f"argument is outside the years Python covers: {value}")
    return moment.replace(tzinfo=None, microsecond=remainder * 1_000_000 // units)


def _split(
    children: list[Any], lengths: pa.Array, build: Callable[[list[Any]], Any]
) -> list[Any]:
    """Splits 'children' into one value per row made by 'build': per row, the
    next 'lengths' children, or None where the length is NULL."""
    values = []
    offset = 0
    for length in lengths.to_pylist():
        if length is None:
            values.append(None)
            continue
        values.append(build(children[offset : offset + length]))
        offset += length
    return values


def _is_named_row(value: Any) -> bool:
    """True for a pyspark Row with field names. A Row exists only once pyspark
    is loaded, so this never imports it."""
    types_module = sys.modules.get("pyspark.sql.types")
    return (
        types_module is not None
        and isinstance(value, types_module.Row)
        and hasattr(value, "__fields__")
    )


def _hashable(value: Any) -> Any:
    """Returns 'value' in a form usable as a dict key: lists become tuples and
    bytearrays become bytes, recursively."""
    if isinstance(value, bytearray):
        return bytes(value)
    if _is_named_row(value):
        make_row = _row_class()(*value.__fields__)
        return make_row(*(_hashable(child) for child in value))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(child) for child in value)
    return value


def _struct_values(array: pa.StructArray, zone: datetime.tzinfo) -> list[Any]:
    row_class = _row_class()
    assert row_class is not None, "_check_readable rejects ROW types without pyspark"
    struct_type = cast(pa.StructType, array.type)
    # Built by position, so empty and repeated field names work.
    make_row = row_class(*(struct_field.name for struct_field in struct_type))
    fields = [_values(array.field(i), zone) for i in range(struct_type.num_fields)]
    valid = cast(pa.BooleanArray, array.is_valid()).to_pylist()
    return [
        make_row(*(values[row] for values in fields)) if valid[row] else None
        for row in range(len(array))
    ]


# Results: Python values to values pyarrow builds the declared type from.
# Converters check the Python type, which pyarrow is lenient about: it
# truncates a float to an integer and splits a str into a list. They also
# check a REAL beyond its range, which pyarrow turns into infinity, and a
# TIMESTAMP more precise than its unit. pyarrow checks the rest of the value:
# an integer's range, an int as a float within ±2^53 (±2^24 for REAL), a
# decimal's precision and scale, UTF-8, a NULL map key. Map keys
# equal in Velox are for the executor to check, with Velox's equality.


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_date(value: Any) -> bool:
    return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)


def _leaf_check(arrow_type: pa.DataType) -> Callable[[Any], bool] | None:
    """Returns a check that a value has a Python type a result of 'arrow_type'
    can have, or None for a type that needs more than a check."""
    if pa.types.is_boolean(arrow_type):
        return lambda value: type(value) is bool
    if _is_signed_integer(arrow_type):
        return _is_integer
    if _is_floating(arrow_type):
        return lambda value: isinstance(value, float) or _is_integer(value)
    if _is_string(arrow_type):
        return lambda value: isinstance(value, str)
    if _is_binary(arrow_type):
        return lambda value: isinstance(value, (bytes, bytearray))
    if pa.types.is_date32(arrow_type):
        return _is_date
    if isinstance(arrow_type, pa.Decimal128Type):
        return lambda value: isinstance(value, decimal.Decimal)
    return None


def _result_converter(
    arrow_type: pa.DataType, zone: datetime.tzinfo
) -> ResultConverter:
    """Returns the converter for results of 'arrow_type'."""
    check_supported(arrow_type)
    declared = _type_name(arrow_type)
    if arrow_type == pa.float32():
        return _real_result(declared)
    check = _leaf_check(arrow_type)
    if check is not None:

        def convert(value: Any) -> Any:
            if not check(value):
                raise _wrong_type(value, declared)
            return value

        return convert
    if pa.types.is_timestamp(arrow_type):
        return _timestamp_result(
            cast(pa.TimestampType, arrow_type).unit, zone, declared
        )
    if _is_list(arrow_type):
        return _list_result(cast(pa.ListType, arrow_type).value_type, zone, declared)
    if pa.types.is_map(arrow_type):
        return _map_result(cast(pa.MapType, arrow_type), zone, declared)
    return _struct_result(cast(pa.StructType, arrow_type), zone, declared)


def _wrong_type(value: Any, declared: str) -> ResultError:
    return ResultError(f"must be {declared}: {type(value).__name__} {value!r}")


def _real_result(declared: str) -> ResultConverter:
    def convert(value: Any) -> Any:
        if not (isinstance(value, float) or _is_integer(value)):
            raise _wrong_type(value, declared)
        if isinstance(value, float) and math.isfinite(value):
            # Some Python versions pack a float beyond REAL's range as
            # infinity, others raise.
            try:
                stored = struct.unpack("f", struct.pack("f", value))[0]
            except OverflowError:
                stored = math.inf
            if math.isinf(stored):
                raise ResultError(f"is out of range for {declared}: {value}")
        return value

    return convert


def _timestamp_result(
    unit: str, zone: datetime.tzinfo, declared: str
) -> ResultConverter:
    units = _UNITS_PER_SECOND[unit]

    def convert(value: Any) -> int:
        if not isinstance(value, datetime.datetime):
            raise _wrong_type(value, declared)
        # A naive datetime is in the session time zone; timestamp() would
        # take it as the host's.
        if value.tzinfo is None:
            value = value.replace(tzinfo=zone)
        seconds = int(value.replace(microsecond=0).timestamp())
        fraction, lost = divmod(value.microsecond * units, 1_000_000)
        if lost:
            raise ResultError(f"is more precise than {declared} in {unit}: {value}")
        return seconds * units + fraction

    return convert


def _list_result(
    element_type: pa.DataType, zone: datetime.tzinfo, declared: str
) -> ResultConverter:
    element = _result_converter(element_type, zone)

    def convert(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)):
            raise _wrong_type(value, declared)
        return [None if child is None else element(child) for child in value]

    return convert


def _map_result(
    map_type: pa.MapType, zone: datetime.tzinfo, declared: str
) -> ResultConverter:
    key = _result_converter(map_type.key_type, zone)
    item = _result_converter(map_type.item_type, zone)

    def convert(value: Any) -> list[tuple[Any, Any]]:
        if not isinstance(value, dict):
            raise _wrong_type(value, declared)
        return [
            (
                None if map_key is None else key(map_key),
                None if map_value is None else item(map_value),
            )
            for map_key, map_value in value.items()
        ]

    return convert


def _struct_result(
    struct_type: pa.StructType, zone: datetime.tzinfo, declared: str
) -> ResultConverter:
    names = [struct_field.name for struct_field in struct_type]
    converters = [
        _result_converter(struct_field.type, zone) for struct_field in struct_type
    ]

    def from_values(values: Sequence[Any]) -> tuple[Any, ...]:
        # A tuple in field order, which pyarrow takes for a struct; unlike a
        # dict, it works with empty and repeated field names.
        return tuple(
            None if child is None else convert(child)
            for convert, child in zip(converters, values)
        )

    def from_dict(value: dict[str, Any]) -> tuple[Any, ...]:
        if len(set(names)) != len(names):
            raise ResultError(
                f"must give fields by position for {declared}, whose field names repeat"
            )
        extra = set(value) - set(names)
        if extra:
            raise ResultError(
                f"has fields that {declared} does not have: "
                f"{', '.join(sorted(map(repr, extra)))}"
            )
        return from_values([value.get(name) for name in names])

    def convert(value: Any) -> tuple[Any, ...]:
        # A Row is read by position when its field names are the declared
        # ones, as for a ROW argument passed back, and by name otherwise.
        if _is_named_row(value) and list(value.__fields__) != names:
            fields = list(value.__fields__)
            if len(set(fields)) != len(fields):
                raise ResultError(
                    f"must not repeat field names other than {declared}'s: {fields}"
                )
            return from_dict(value.asDict())
        if isinstance(value, dict):
            return from_dict(value)
        if not isinstance(value, tuple):
            raise _wrong_type(value, declared)
        if len(value) != len(names):
            raise ResultError(
                f"must have {len(names)} values for {declared}: {len(value)}"
            )
        return from_values(value)

    return convert


_BUILD_ERRORS = (pa.ArrowException, ValueError, TypeError, OverflowError)


def _array(values: list[Any], arrow_type: pa.DataType) -> pa.Array:
    array = pa.array(values, arrow_type)
    if isinstance(array, pa.ChunkedArray):
        raise ValueError("results exceed the size of one Arrow array")
    return array


def _build_array(
    values: list[Any], errors: list[str | None], function: "Function"
) -> pa.Array:
    """Returns an array of the function's result type from converted results.
    A value pyarrow rejects becomes NULL, and its row's error says why."""
    declared = _type_name(function.result_type)
    storage_type = _timestamps_as_int64(function.result_type)
    try:
        array = _array(values, storage_type)
    except _BUILD_ERRORS:
        for row, value in enumerate(values):
            try:
                _array([value], storage_type)
            except _BUILD_ERRORS as e:
                values[row] = None
                errors[row] = _utf8(
                    f"{function.name} result does not fit {declared}: {e}"
                )
        try:
            array = _array(values, storage_type)
        except _BUILD_ERRORS as e:
            # Every row fits alone, but not all together.
            message = _utf8(f"{function.name} result does not fit {declared}: {e}")
            for row, error in enumerate(errors):
                if error is None:
                    errors[row] = message
            array = pa.nulls(len(values), storage_type)
    # pyarrow's type equality ignores list and map field names, which the
    # cast restores; compare the types' full spellings.
    if str(array.type) == str(function.result_type):
        return array
    return array.cast(function.result_type)


# Functions and batches.


@dataclass(frozen=True)
class Argument:
    """A column of BATCH, or the results of an earlier function."""

    column: int | None
    function: int | None


@dataclass(frozen=True)
class Function:
    """A function to call, its arguments, and its result type."""

    name: str
    callable: Callable[..., Any]
    arguments: list[Argument]
    result_type: pa.DataType
    convert_result: ResultConverter
    # Whether RESULT returns this function's results; otherwise only later
    # functions use them.
    output: bool


@dataclass(frozen=True)
class Session:
    """The functions to run, the session time zone, and how long unpickling
    the functions took."""

    functions: list[Function]
    zone: datetime.tzinfo
    unpickle_seconds: float


def _require(spec: dict[str, Any], key: str, where: str) -> Any:
    """Returns spec[key], failing for a missing key; 'where' names the spec."""
    if key not in spec:
        raise ProtocolError(f"{where} must have key: {key}")
    return spec[key]


def _parse_argument(spec: Any, function_index: int) -> Argument:
    """Returns the argument 'spec' describes for the function at
    'function_index': {"column": n} or {"function": n}."""
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ProtocolError(f"Argument must be a column or a function: {spec}")
    if _is_integer(spec.get("column")):
        if spec["column"] < 0:
            raise ProtocolError(f"Argument column must not be negative: {spec}")
        return Argument(column=spec["column"], function=None)
    if _is_integer(spec.get("function")):
        if not 0 <= spec["function"] < function_index:
            raise ProtocolError(
                f"Function argument must refer to an earlier function: {spec}"
            )
        return Argument(column=None, function=spec["function"])
    raise ProtocolError(f"Argument must be a column or a function: {spec}")


def setup(payload: bytes) -> Session:
    """Returns the session SETUP describes."""
    sections = split_sections(payload)
    if len(sections) < 2:
        raise ProtocolError("SETUP must have a header and a result schema")
    header = json.loads(sections[0])
    version = _require(header, "protocolVersion", "SETUP header")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"Protocol version must be {PROTOCOL_VERSION}: {version}")
    pickled_with = _require(header, "pythonVersion", "SETUP header")
    worker_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if pickled_with != worker_version:
        raise ProtocolError(
            f"Functions must be pickled with the worker's Python {worker_version}: "
            f"{pickled_with}"
        )
    zone = parse_time_zone(_require(header, "sessionTimeZone", "SETUP header"))
    specs = _require(header, "functions", "SETUP header")
    schema = pa.ipc.read_schema(pa.py_buffer(sections[1]))
    pickled = sections[2:]
    if not len(specs) == len(schema) == len(pickled):
        raise ProtocolError(
            "SETUP must have as many result types and pickled functions as "
            f"functions: {len(specs)}, {len(schema)}, {len(pickled)}"
        )
    start = time.perf_counter()
    functions = [
        _load_function(index, spec, schema_field.type, code, zone)
        for index, (spec, schema_field, code) in enumerate(zip(specs, schema, pickled))
    ]
    if not any(function.output for function in functions):
        raise ProtocolError("SETUP must have at least one output function")
    for function in functions:
        for argument in function.arguments:
            if argument.function is not None:
                _check_readable(functions[argument.function].result_type)
    return Session(functions, zone, time.perf_counter() - start)


def _load_function(
    index: int,
    spec: dict[str, Any],
    result_type: pa.DataType,
    code: bytes,
    zone: datetime.tzinfo,
) -> Function:
    """Returns the function at 'index' in SETUP: its 'spec' from the header,
    its result type, and its pickled 'code'."""
    name = _require(spec, "name", "Function")
    arguments = [
        _parse_argument(argument, index)
        for argument in _require(spec, "arguments", "Function")
    ]
    output = spec.get("output", True)
    if not isinstance(output, bool):
        raise ProtocolError(f"Function output must be true or false: {name}")
    try:
        # Running the code the client shipped is the worker's purpose.
        # @lint-ignore PYTHONPICKLEISBAD
        loaded = pickle.loads(code)
    # Importing the function's modules can raise anything, SystemExit included.
    except BaseException as e:  # noqa: B036
        raise ProtocolError(
            f"Pickled function failed to unpickle: {name}: {type(e).__name__}: {e}"
        ) from e
    # A pickled function is a callable, or a tuple whose first item is the
    # callable (README.md).
    callable_ = loaded[0] if isinstance(loaded, tuple) and loaded else loaded
    if not callable(callable_):
        raise ProtocolError(f"Pickled function must be callable: {name}")
    return Function(
        name=name,
        callable=callable_,
        arguments=arguments,
        result_type=result_type,
        convert_result=_result_converter(result_type, zone),
        output=output,
    )


@dataclass
class Outcome:
    """A function's results on a batch: the result array, per row an error or
    None, the results as later functions receive them, if any do, and the wall
    time spent calling the function, building its results and converting them
    for later functions."""

    results: pa.Array
    errors: list[str | None]
    argument_values: list[Any] | None
    seconds: float


def _argument_values(
    function: Function, columns: dict[int, list[Any]], outcomes: list[Outcome]
) -> list[list[Any]]:
    """Returns the values of each argument of 'function', one list per
    argument. An earlier function's results reach it as they would through
    Velox, and its failed rows as Unconvertible."""
    values = []
    for argument in function.arguments:
        if argument.column is not None:
            values.append(columns[argument.column])
            continue
        inner_values = outcomes[cast(int, argument.function)].argument_values
        assert inner_values is not None, "run_batch converts results functions read"
        values.append(inner_values)
    return values


def _result_values(
    function: Function,
    results: pa.Array,
    errors: list[str | None],
    zone: datetime.tzinfo,
) -> list[Any]:
    """Returns 'function''s results as later functions receive them: as they
    would arrive through Velox, and its failed rows as Unconvertible."""
    return [
        value
        if error is None
        else Unconvertible(f"input {function.name} failed: {error}")
        for value, error in zip(_column_values(results, zone), errors)
    ]


def _call_row(
    function: Function, row_arguments: list[Any], with_traceback: bool
) -> tuple[Any, str | None, bool]:
    """Calls 'function' on one row. Returns its converted result, its error or
    None, and whether it raised; 'with_traceback' adds the traceback to a
    raised error."""
    failed = next(
        (argument for argument in row_arguments if isinstance(argument, Unconvertible)),
        None,
    )
    if failed is not None:
        return None, _utf8(f"{function.name} {failed.reason}"), False
    try:
        value = function.callable(*row_arguments)
        if value is None:
            return None, None, False
        return function.convert_result(value), None, False
    except ResultError as e:
        return None, _utf8(f"{function.name} result {e}"), False
    # User code can raise anything, sys.exit's SystemExit included; it fails
    # the row.
    except BaseException as e:  # noqa: B036
        message = f"{function.name} raised {_describe(e)}"
        if with_traceback:
            message += "\n" + traceback.format_exc()
        return None, _utf8(message), True


def _describe(e: BaseException) -> str:
    """Returns the exception's type and message. The message is user code,
    which can itself raise."""
    try:
        return f"{type(e).__name__}: {e}"
    except BaseException:  # noqa: B036
        return f"{type(e).__name__}: <message failed>"


def _utf8(text: str) -> str:
    """Returns 'text' with what UTF-8 cannot encode, such as lone surrogates,
    escaped."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _call_function(
    function: Function, arguments: list[list[Any]], num_rows: int
) -> tuple[pa.Array, list[str | None]]:
    """Calls 'function' on each row; returns its results and per row an error
    or None. A row that raises, receives an Unconvertible argument, or returns
    a value of the wrong type fails alone. The function's first row that
    raises carries the traceback."""
    converted: list[Any] = [None] * num_rows
    errors: list[str | None] = [None] * num_rows
    traced = False
    for row in range(num_rows):
        converted[row], errors[row], raised = _call_row(
            function, [values[row] for values in arguments], not traced
        )
        traced = traced or raised
    return _build_array(converted, errors, function), errors


def _check_columns(function: Function, num_columns: int) -> None:
    for argument in function.arguments:
        if argument.column is not None and not 0 <= argument.column < num_columns:
            raise ProtocolError(
                f"Argument column must be one of BATCH's {num_columns}: "
                f"{argument.column}"
            )


def run_batch(session: Session, payload: bytes) -> bytes:
    """Returns the RESULT payload for a BATCH payload."""
    batch = read_batch(payload)
    for function in session.functions:
        _check_columns(function, batch.num_columns)
    arguments = [
        argument for function in session.functions for argument in function.arguments
    ]
    read = sorted(
        {argument.column for argument in arguments if argument.column is not None}
    )
    read_functions = {
        argument.function for argument in arguments if argument.function is not None
    }
    for index in read:
        check_supported(batch.column(index).type)
        _check_readable(batch.column(index).type)
    columns = {
        index: _column_values(batch.column(index), session.zone) for index in read
    }
    outcomes: list[Outcome] = []
    for index, function in enumerate(session.functions):
        start = time.perf_counter()
        results, errors = _call_function(
            function, _argument_values(function, columns, outcomes), batch.num_rows
        )
        argument_values = (
            _result_values(function, results, errors, session.zone)
            if index in read_functions
            else None
        )
        outcomes.append(
            Outcome(results, errors, argument_values, time.perf_counter() - start)
        )
    return write_batch(_result_batch(session.functions, outcomes))


def _result_batch(functions: list[Function], outcomes: list[Outcome]) -> pa.RecordBatch:
    """Returns the RESULT batch: the output functions' columns, and every
    function's time in the metadata."""
    arrays: list[Any] = []
    for function, outcome in zip(functions, outcomes):
        if function.output:
            arrays += [outcome.results, pa.array(outcome.errors, pa.string())]
    wall_seconds = json.dumps([outcome.seconds for outcome in outcomes])
    return pa.record_batch(arrays, names=[""] * len(arrays)).replace_schema_metadata(
        {"functionWallSeconds": wall_seconds}
    )


# Serving.


def serve(stream: io.BufferedIOBase) -> None:
    """Serves one session: SETUP, then BATCH messages until END."""
    message_type, payload = read_message(stream)
    if message_type != SETUP:
        raise ProtocolError(f"First message must be SETUP: {message_type}")
    session = setup(payload)
    write_message(
        stream,
        READY,
        json.dumps({"unpickleSeconds": session.unpickle_seconds}).encode(),
    )
    while True:
        message_type, payload = read_message(stream)
        if message_type == BATCH:
            write_message(stream, RESULT, run_batch(session, payload))
        elif message_type == END:
            write_message(stream, END, b"")
            return
        else:
            raise ProtocolError(f"Message must be BATCH or END: {message_type}")


def run(stream: io.BufferedIOBase) -> int:
    """Serves the executor on 'stream' and reports any failure outside a row as
    ERROR. Returns the process exit status."""
    try:
        serve(stream)
    # Whatever ends the session, the executor must get ERROR.
    except BaseException as e:  # noqa: B036
        error = {
            "message": _utf8(_describe(e)),
            "traceback": _utf8(traceback.format_exc()),
        }
        try:
            write_message(stream, ERROR, json.dumps(error).encode())
        except OSError:
            pass
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fd", type=int, required=True, help="Inherited socket")
    args = parser.parse_args(argv)
    connection = socket.socket(fileno=args.fd)
    stream = connection.makefile("rwb")
    try:
        return run(stream)
    finally:
        stream.close()
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
