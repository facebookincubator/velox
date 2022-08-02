import cpp_micro_benchmarks


def test_parse_benchmark_name_no_params():
    tags = cpp_micro_benchmarks._parse_benchmark_name("Something")
    assert tags == {"name": "Something"}


def test_get_values():
    result = [
        "/some/path/FeatureNormalization.cpp",
        "normalize",
        20.99487392089844,
    ]
    benchmark = cpp_micro_benchmarks.RecordCppMicroBenchmarks()
    actual = benchmark._get_values(result)
    assert actual == {
        "data": [20.99487392089844],
        "time_unit": "s",
        "times": [20.99487392089844],
        "unit": "s",
    }


def test_format_unit():
    benchmark = cpp_micro_benchmarks.RecordCppMicroBenchmarks()
    assert benchmark._format_unit("bytes_per_second") == "B/s"
    assert benchmark._format_unit("items_per_second") == "i/s"
    assert benchmark._format_unit("foo_per_bar") == "foo_per_bar"


def test_get_run_command():
    options = {
        "iterations": None,
    }
    actual = cpp_micro_benchmarks.get_run_command("out", options)
    assert actual == [
        "../../../scripts/benchmark-runner.py",
        "run",
        "--path",
        "../../../_build/release/velox/benchmarks/basic/",
        "--dump-path",
        "out",
    ]
