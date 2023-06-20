from cmake_refactor import io
import os
import tempfile
import pytest 

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)


def test_file_roundtrip():
    original = os.path.join(current_dir, "CMakeLists.txt")
    stream = io.get_token_stream(original)

    with tempfile.NamedTemporaryFile() as new:
        io.write_token_stream(new.name, stream)
        new_contents = new.read().decode('utf-8')

    with open(original, 'r', encoding='utf-8') as orig:
        original_contents = orig.read()

    assert original_contents.splitlines() == new_contents.splitlines()
