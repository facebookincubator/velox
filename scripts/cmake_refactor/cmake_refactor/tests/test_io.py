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
    files = io.find_files(file, repo,  ['type_calculation', 'experimental'])
    targets = {}

    for f in files:
        # print("Parsing: ", f)
        io.parse_targets(f, targets)
    print(len(targets))
    print(targets['velox_common_base'])
    print(targets['velox_flag_definitions'])
    assert targets['velox_flag_definitions'].is_object_lib
    # print(targets['velox_common_base'])
    # for target in targets:
    #     print(targets[target].name)


def test_get_includes():
    file = os.path.join(current_dir, "BitUtil.cpp")
    incs = io.get_includes(file)
    assert len(incs[0]) == 3
    assert len(incs[1]) == 2


def test_get_dep():
    assert io.get_dep_name('snappy.h') == 'Snappy::snappy'
    assert io.get_dep_name(
        'thrift/protovol/TCompactProtocol.h') == 'thrift::thrift'
    assert io.get_dep_name(
        'folly/synchronization/AtomicStruct.h') == 'Folly::folly'
    assert io.get_dep_name('glog/glog.h') == 'glog::glog'


def test_header_map():
    repo_root = '/home/jwj/code/velox/'
    file = "CMakeLists.txt"
    repo = os.path.abspath(os.path.join(current_dir, '../../../../velox'))
    files = io.find_files(
        file, repo,  ['experimental'])
    targets = {}
    hm = {}
    for f in files:
        # print("Parsing: ", f)
        io.parse_targets(f, targets, header_target_map=hm, repo_root=repo_root)
    io.map_local_headers(targets, hm, repo_root)
    assert targets.get('velox_common_base') is not None
    assert hm.get("velox/common/base/VeloxException.h") is not None


def test_full_update():
    io.update_links('velox', '/home/jwj/code/velox/', ['experimental'])
