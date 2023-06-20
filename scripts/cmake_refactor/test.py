import sys

from antlr4 import *
from antlr4.Token import CommonToken
from antlr4.TokenStreamRewriter import TokenStreamRewriter

from parser.CMakeLexer import CMakeLexer
from parser.CMakeListener import CMakeListener
from parser.CMakeParser import CMakeParser


def pretty_print_listener_context(ctx, indent=0):
    indent_str = " " * indent
    print(f"{indent_str}{type(ctx).__name__}: {ctx.getText()}")

    if ctx.getChildCount() > 0:
        for child_ctx in ctx.getChildren():
            pretty_print_listener_context(child_ctx, indent + 2)


class Printer(CMakeListener):
    def __init__(self, token_stream):
        super().__init__()
        self.tokens = TokenStreamRewriter(token_stream)

    def exitTarget_command(self, ctx):
        if ctx.Target_command().getText() == 'add_library':
            start_index = ctx.stop.tokenIndex
            target = ctx.alias().getText()
            self.tokens.insertAfter(
                start_index, f'\nadd_library(velox::{target.removeprefix("velox_")} INTERFACE {target})\n')
            pretty_print_listener_context(ctx)

            if ctx.keyword() is not None and ctx.keyword().public():
                print(ctx.keyword().getText())


def main(argv):
    input_stream = FileStream(argv[1])
    lexer = CMakeLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = CMakeParser(stream)
    tree = parser.file_()
    walker = ParseTreeWalker()
    li = Printer(stream)
    walker.walk(li, tree)
    new_text = li.tokens.getText('default', 0, 999999999)
    # print(new_text)
    with open(f'new_{argv[1]}', 'w') as file:
        file.write(new_text)

if __name__ == '__main__':
    main(sys.argv)
