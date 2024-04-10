import uuid
from os.path import splitext
from pathlib import Path
from typing import Any, Dict, List

from benchadapt import BenchmarkResult
from benchadapt.adapters import BenchmarkAdapter


class BinarySizeAdapter(BenchmarkAdapter):
    size_file: Path

    def __init__(
        self,
        command: List[str],
        # TODO remove default path
        size_file: str = "/tmp/object-size",
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
                # TODO remove, added via envvars
                github={
                    "repository": "https://github.com/facebookincubator/velox",
                    "commit": "e29cde7b220ac507f7c55e3101f47836a67c51f1",
                },
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
    ninja_log: Path

    def __init__(
        self,
        command: List[str],
        # TODO remove default path
        ninja_log: str = "_build/release/.ninja_log",
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
                # TODO remove, added via envvars
                github={
                    "repository": "https://github.com/facebookincubator/velox",
                    "commit": "c329af5d37547f8ab3e88129b7aa166294e9d75c",
                },
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
                # TODO remove, added via envvars
                github={
                    "repository": "https://github.com/facebookincubator/velox",
                    "commit": "c329af5d37547f8ab3e88129b7aa166294e9d75c",
                },
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
