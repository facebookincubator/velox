import antlr4 as ant
import os
from .parser.CMakeLexer import CMakeLexer
from .parser.CMakeParser import CMakeParser
from .parser.CMakeListener import CMakeListener
from . import listeners


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


def find_files(file_name: str, root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ['type_calculation', 'experimental']]
        for file in files:
            if file == file_name:
                file_paths.append(os.path.join(root, file))
    return file_paths


def parse_targets(file: str, targets: dict[str, listeners.TargetNode]):
    stream = get_token_stream(file)
    listener = listeners.TargetInputListener(targets)
    walk_stream(stream, listener)
    return targets
