from cmake_refactor import io, listeners
import os
import tempfile


current_dir = os.path.dirname(os.path.abspath(__file__))
cml = os.path.join(current_dir, "CMakeLists.txt")


def test_file_roundtrip():
    original = cml
    stream = io.get_token_stream(original)

    with tempfile.NamedTemporaryFile() as new:
        io.write_token_stream(new.name, stream)
        new_contents = new.read().decode("utf-8")

    with open(original, "r", encoding="utf-8") as orig:
        original_contents = orig.read()

    assert original_contents.splitlines() == new_contents.splitlines()


class Listener_test(io.CMakeListener):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def exitAdd_library(self, ctx: io.CMakeParser.Add_libraryContext):
        self.counter += 1
        print("Found target:", ctx.target().getText())


def test_listener():
    stream = io.get_token_stream(cml)
    listener = Listener_test()
    io.walk_stream(stream, listener)
    assert listener.counter == 2


def test_find_file():
    file = "CMakeLists.txt"
    files = io.find_files(file, current_dir)
    assert len(files) == 1


def test_parse_dir():
    file = "CMakeLists.txt"
    repo = os.path.abspath(os.path.join(current_dir, '../../../../velox'))
    files = io.find_files(file, repo)
    targets = {}

    for f in files:
        print("Parsing: ", f)
        io.parse_targets(f, targets)
    print(len(targets))
    assert False
