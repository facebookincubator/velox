import uuid
from os.path import splitext
from pathlib import Path
from typing import Any, Dict, List

from benchadapt import BenchmarkResult
from benchadapt.adapters import BenchmarkAdapter


class BinarySizeAdapter(BenchmarkAdapter):
    """
    Adapter to track build artifact sizes in conbench.
    Expects the `size_file` to be formatted like this:
    <size in bytes> <path/to/binary|arbitrary_name>

    Suite meta data will be library, object or executable
    based on file ending.
    """

    size_file: Path

    def __init__(
        self,
        command: List[str],
        size_file: str,
        result_fields_override: Dict[str, Any] = {},
        result_fields_append: Dict[str, Any] = {},
    ) -> None:
        self.size_file = Path(size_file)
        super().__init__(command, result_fields_override, result_fields_append)

    def _transform_results(self) -> List[BenchmarkResult]:
        results = []

        batch_id = uuid.uuid4().hex
        with open(self.size_file, "r") as file:
            sizes = [line.strip() for line in file]

        if not sizes:
            raise ValueError("'size_file' is empty!")

        for line in sizes:
            size, path = line.split(maxsplit=1)
            path = path.strip()
            _, ext = splitext(path)
            if ext in [".so", ".a"]:
                suite = "library"
            elif ext == ".o":
                suite = "object"
            else:
                suite = "executable"

            parsed_size = BenchmarkResult(
                run_reason="merge",
                batch_id=batch_id,
                stats={
                    "data": [size],
                    "unit": "B",
                    "iterations": 1,
                },
                tags={"name": path, "suite": suite, "source": "build_metrics"},
                info={},
                context={"benchmark_language": "C++"},
            )
            results.append(parsed_size)

        return results


class BuildTimeAdapter(BenchmarkAdapter):
    """
    Adapter to extract compile and link times from a .ninja_log.
    Will calculate aggregates for total, compile, link and wall time.
    Suite metadata will be set based on binary ending to object, library or executable.

    Only files in paths beginning with velox/ will be tracked to avoid dependencies.
    """

    ninja_log: Path

    def __init__(
        self,
        command: List[str],
        ninja_log: str,
        result_fields_override: Dict[str, Any] = {},
        result_fields_append: Dict[str, Any] = {},
    ) -> None:
        self.ninja_log = Path(ninja_log)
        super().__init__(command, result_fields_override, result_fields_append)

    def _transform_results(self) -> List[BenchmarkResult]:
        results = []

        batch_id = uuid.uuid4().hex
        with open(self.ninja_log, "r") as file:
            log_lines = [line.strip() for line in file]

        if not log_lines[0].startswith("# ninja log v"):
            raise ValueError("Malformed Ninja log found!")
        else:
            del log_lines[0]

        ms2sec = lambda x: x / 1000
        get_epoch = lambda l: int(l.split()[2])
        totals = {
            "link_time": 0,
            "compile_time": 0,
            "total_time": 0,
            "wall_time": get_epoch(log_lines[-1]) - get_epoch(log_lines[0]),
        }

        for line in log_lines:
            start, end, epoch, object_path, _ = line.split()
            start = int(start)
            end = int(end)
            duration = ms2sec(end - start)

            # Don't track dependency times (refine check potentially?)
            if not object_path.startswith("velox"):
                continue

            _, ext = splitext(object_path)
            if ext in [".so", ".a"] or not ext:
                totals["link_time"] += duration
                suite = "linking"
            elif ext == ".o":
                totals["compile_time"] += duration
                suite = "compiling"
            else:
                print(f"Unkown file type found: {object_path}")
                print("Skipping...")
                continue

            time_result = BenchmarkResult(
                run_reason="merge",
                batch_id=batch_id,
                stats={
                    "data": [duration],
                    "unit": "s",
                    "iterations": 1,
                },
                tags={"name": object_path, "suite": suite, "source": "build_metrics"},
                info={},
                context={"benchmark_language": "C++"},
            )
            results.append(time_result)

        totals["total_time"] = totals["link_time"] + totals["compile_time"]
        for total_name, total in totals.items():
            total_result = BenchmarkResult(
                run_reason="merge",
                batch_id=batch_id,
                stats={
                    "data": [total],
                    "unit": "s",
                    "iterations": 1,
                },
                tags={"name": total_name, "suite": "total", "source": "build_metrics"},
                info={},
                context={"benchmark_language": "C++"},
            )
            results.append(total_result)

        return results


# find velox -type f -name '*.o' -exec ls -l -BB {} \; | awk '{print $5, $9}' |  sed 's|CMakeFiles/.*dir/||g' > /tmp/object-size
BuildTimeAdapter(command=["true"])()
# BinarySizeAdapter(command=["true"])()
