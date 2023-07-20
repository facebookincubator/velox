import antlr4 as ant
import os
from .parser.CMakeLexer import CMakeLexer
from .parser.CMakeParser import CMakeParser
from .parser.CMakeListener import CMakeListener
from . import listeners
import re
from glob import glob


def get_token_stream(file_path: str) -> ant.CommonTokenStream:
    input_stream = ant.FileStream(file_path)
    lexer = CMakeLexer(input_stream)
    return ant.CommonTokenStream(lexer)


def write_token_stream(file_path: str, stream: ant.CommonTokenStream) -> None:
    # TODO find better way to get entire text
    text = stream.getText(0, 999999999)
    with open(file_path, "w") as file:
        file.write(text)


def walk_stream(stream: ant.CommonTokenStream, listener: CMakeListener):
    parser = CMakeParser(stream)
    parser.addErrorListener(listeners.SyntaxErrorListener())
    tree = parser.file_()
    walker = ant.ParseTreeWalker()
    walker.walk(listener, tree)
    return listener


def find_files(file_name: str, root_dir, excluded_dirs: list[str] = []):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        file_paths.extend([os.path.join(root, f)
                          for f in files if file_name in f])
    return file_paths


def parse_targets(file: str, targets: dict[str, listeners.TargetNode],
                  header_target_map=None, repo_root=''):
    stream = get_token_stream(file)
    listener = listeners.TargetInputListener(
        targets, header_target_map=header_target_map, repo_root=repo_root)
    walk_stream(stream, listener)
    return targets


def get_includes(file_path: str) -> tuple[list[str], list[str]]:
    include_ptrn = re.compile(
        "^#include\\s+[\"<]([\\w/]+\\.h.*)[\">]",
        flags=re.IGNORECASE | re.MULTILINE)
    with open(file_path, "r") as file:
        src = file.read()
        includes: list[str] = re.findall(include_ptrn, src)

    return ([h for h in includes if h.startswith('velox')],
            [h for h in includes if not h.startswith('velox')])


def map_local_headers(targets: dict[str, listeners.TargetNode],
                      header_target_map: dict[str, [listeners.TargetNode]], repo_root: str):
    # header:[dependency targets]
    # todo seperste map for header cpp matching?
    # e.g. for a header with no cpp file
    # velox/common/base/BloomFilter.h: [velox_common_base, velox_exception]
    # This means any target using the header needs to link against these targets.
    # Header with no dependencies e.g. common/base/IOUtils.h:[]
    # We only care for headers used in cpp files/headers including these
    # as due to the global include dirs there are no header only targets.
    def resolve_includes(files: list[str],
                         target_list: list[listeners.TargetNode]):
        cpp_incs = []
        for file in files:
            velox_h, deps_h = get_includes(file)
            cwd = os.path.dirname(file)
            local_h = glob('*.h*', root_dir=cwd)
            # handle local headers used without full include path
            no_path_h = [h for h in deps_h if h in local_h]
            deps_h = [h for h in deps_h if h not in no_path_h]
            no_path_h = [os.path.join(cwd.removeprefix(
                repo_root), h) for h in no_path_h]
            # don't parse ddb headers to avoid issues with vendored deps and
            # C stdlib headers
            if target.name not in ['duckdb', 'tpch_extension', 'dbgen']:
                dependencies = [d for h in deps_h if (d := get_dep_name(h))]
                for dep in dependencies:
                    dep_target = targets.get(dep, listeners.TargetNode(dep))

                    # We can directly add these dependencies as targets as
                    # we already know which header belongs to which target.
                    if dep_target not in target_list:
                        Warning(
                            f'New dependency {dep_target.name} added to target {target.name}')
                        target_list.append(dep_target)

            cpp_incs.extend([h for h in velox_h if h not in target_h])
        return cpp_incs

    for key, target in targets.items():
        if target.cml_path is None:
            continue

        target_h = [h.removeprefix(repo_root) for h in target.headers]

        target.cpp_includes = [
            *set(resolve_includes(target.sources, target.private_targets))]
        target.h_includes = [
            *set(resolve_includes(target.headers, target.public_targets))]

    # have to do second pass to avoid mixups
    for key, target in targets.items():
        if target.cml_path is None:
            continue
        target.private_targets.extend(
            [t for h in target.cpp_includes if (t := header_target_map.get(h)) is not None])
        target.public_targets.extend(
            [t for h in target.h_includes if (t := header_target_map.get(h)) is not None])

    return header_target_map


def get_dep_name(header: str) -> str:
    name = os.path.splitext(header.lower())[0].split('/')
    header_libs = ['xxhash', 'fcntl', 'linux', 'sys',
                   'limits', 'date', 'time', 'pthread', 'glob']
    if name[0] in header_libs or 'intrin' in name[0] or ('std' in name[0] and not name[0] == 'zstd'):
        name[0] = 'header lib'
    match name[0]:
        case 'arrow':
            target = 'arrow'
        case 'aws':
            # todo find a better way...
            target = '${AWSSDK_LIBRARIES}'
        case 'hdfs':
            target = '${LIBHDFS3}'
        case 'boost':
            boost_header = [
                "algorithm",
                "crc",
                "circular_buffer",
                'lexical_cast',
                "math",
                "multi_index",
                "numeric",
                "process",
                "random",
                "uuid",
                "variant"
            ]
            if name[1] in boost_header:
                target = 'Boost::headers'
            else:
                target = f'Boost::{name[1]}'
        case 'folly':
            target = 'Folly::folly'
        case 'fmt':
            target = 'fmt::fmt'
        case 'gflags':
            target = 'gflags::gflags'
        case 'gtest':
            # gtest or gtest_main? or rather Gtest::main?
            target = 'gtest'
        case 'gmock':
            target = 'gmock'
        case 'glog':
            target = 'glog::glog'
        case 'lz4':
            target = 'lz4::lz4'
        case 'parquet':
            target = 'parquet'
        case 're2':
            target = 're2::re2'
        case 'snappy':
            target = 'Snappy::snappy'
        case 'thrift':
            target = 'thrift::thrift'
        case 'xsimd':
            target = 'xsimd'
        case 'zstd':
            target = 'zstd::zstd'
        case 'zlib':
            target = 'ZLIB::ZLIB'
        case 'header lib':
            target = ''
        case _:
            raise Exception(
                f'Found unmatched dependency {name[0]} with header {header}')

    return target


def no_ext_path(path, root=''):
    return os.path.splitext(path.removeprefix(root))[0]


def has_matching_src(header: str, sources: list[str], repo_root='') -> bool:
    header = no_ext_path(header, repo_root)
    for src in sources:
        if header == no_ext_path(src, repo_root):
            return True

    return False


def update_links(src_dir: str, repo_root: str, excluded_dirs: list[str] = [],
                 dry_run=True):
    file = "CMakeLists.txt"
    repo_root = os.path.abspath(repo_root)
    files = find_files(file, os.path.join(repo_root, src_dir), excluded_dirs)
    targets: dict[str, listeners.TargetNode] = {}
    hm: dict[str, listeners.TargetNode] = {}
    for f in files:
        print(f"Parsing: {f}")
        parse_targets(f, targets, header_target_map=hm, repo_root=repo_root)
    print("Building Dependency Tree")
    map_local_headers(targets, hm, repo_root)

    for t in targets.values():
        t.was_linked = False

    for f in files:

        print(f"Parsing: {f}")
        token_stream = get_token_stream(f)
        update_listener = listeners.UpdateTargetsListener(
            targets, token_stream)
        walk_stream(token_stream, update_listener)
        updated_cml = update_listener.token_stream.getText(
            'default', 0, 999999999)
        if not dry_run:
            with open(f, 'w') as new_f:
                new_f.write(updated_cml)
