from cmake_refactor import io, listeners
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cml = os.path.join(current_dir, "CMakeLists.txt")


def test_source_listener():
    stream = io.get_token_stream(cml)
    targets: dict[str, listeners.TargetNode] = {}
    listener = listeners.TargetInputListener(targets)
    io.walk_stream(stream, listener)
    assert len(targets) == 11


def test_link_listener():
    stream = io.get_token_stream(cml)
    targets: dict[str, listeners.TargetNode] = {}
    listener = listeners.TargetInputListener(targets)
    io.walk_stream(stream, listener)
    velox_exception_public = ['velox_flag_definitions', 'velox_process',
                              'glog::glog',
                              'Folly::folly',
                              'fmt::fmt',
                              'gflags::gflags']
    velox_common_base_public = ['velox_exception']
    velox_common_base_private = ['velox_process', 'xsimd']

    assert velox_exception_public == [
        f.name for f in targets['velox_exception'].public_targets]
    assert velox_common_base_public == [
        f.name for f in targets['velox_common_base'].public_targets]
    assert velox_common_base_private == [
        f.name for f in targets['velox_common_base'].private_targets]


def test_alias_listener():
    stream = io.get_token_stream(cml)
    targets: dict[str, listeners.TargetNode] = {}
    listener = listeners.TargetInputListener(targets)
    io.walk_stream(stream, listener)
    assert targets['velox::exception'].alias_for.name == 'velox_exception'


def test_interface_listener():
    stream = io.get_token_stream(cml)
    targets: dict[str, listeners.TargetNode] = {}
    listener = listeners.TargetInputListener(targets)
    io.walk_stream(stream, listener)
