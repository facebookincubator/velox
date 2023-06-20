import antlr4 as ant
from .parser.CMakeLexer import CMakeLexer
from .parser.CMakeParser import CMakeParser


def get_token_stream(file_path: str) -> ant.CommonTokenStream:
    input_stream = ant.FileStream(file_path)
    lexer = CMakeLexer(input_stream)
    return ant.CommonTokenStream(lexer)


def write_token_stream(file_path: str, stream: ant.CommonTokenStream) -> None:
    # TODO find better way to get entire text
    text = stream.getText(0, 999999999)
    with open(file_path, "w") as file:
        file.write(text)
