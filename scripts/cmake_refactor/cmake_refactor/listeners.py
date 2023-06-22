from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import CancellationException
from .parser.CMakeListener import CMakeListener
from .parser.CMakeParser import CMakeParser
import os


class TargetNode:
    def __init__(
            self, name: str, headers: list[str] = [], sources: list[str] = [],
            is_interface: bool = False, alias_for=None
    ) -> None:
        self.name: str = name
        self.headers: list[str] = headers
        self.public_headers: list[str] = []
        self.sources: list[str] = sources
        self.public_targets: list[TargetNode] = []
        self.private_targets: list[TargetNode] = []
        self.is_interface = is_interface
        self.alias_for: TargetNode = alias_for


class SyntaxErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise CancellationException(f"line {line}:{column} {msg}")


class TargetInputListener(CMakeListener):
    def __init__(self, target_dict: dict[str, TargetNode]) -> None:
        super().__init__()
        self.targets = target_dict

    def get_abs_path(self, files: list[str], ctx):

        # TODO add check that sources actually exist
        return files

    def add_linked_targets(self, node: TargetNode,
                           keyword: CMakeParser.KeywordContext,
                           targets: CMakeParser.TargetContext):
        targets = [t.getText() for t in targets]
        nodes: list[TargetNode] = []

        for target in targets:
            target_node: TargetNode = self.targets.get(target)
            if target_node is None:
                target_node = TargetNode(target)
                self.targets[target] = target_node
            nodes.append(target_node)

        if keyword is None or keyword.public() or keyword.interface():
            node.public_targets.extend(nodes)
        else:
            node.private_targets.extend(nodes)

    def exitAdd_library(self, ctx: CMakeParser.Add_libraryContext):
        target = ctx.target().getText()

        files = [file.getText() for file in ctx.source_file()]
        files = self.get_abs_path(files, ctx)

        base_path = os.path.dirname(ctx.start.getInputStream().fileName)
        files = [f.replace('${CMAKE_CURRENT_LIST_DIR}/', base_path)
                 for f in files]
        # TODO filter out cmake vars
        files = [os.path.join(base_path, f)
                 for f in files if not os.path.dirname(f)]

        headers = [f for f in files if os.path.splitext(f)[1] in [
            ".h", ".hpp"]]
        sources = [f for f in files if os.path.splitext(
            f)[1] in [".c", ".cpp", ".cxx", ".cc", ".c++"]]

        node = self.targets.get(target)
        if node is None:
            self.targets[target] = TargetNode(target, headers, sources)
        else:
            node.sources.extend(sources)
            node.headers.extend(headers)

    def exitLink_libraries(self, ctx: CMakeParser.Link_librariesContext):
        target: str = ctx.target().getText()
        node: TargetNode = self.targets.get(target)
        if node is None:
            node = TargetNode(target)
            self.targets[target] = node

        self.add_linked_targets(node, ctx.keyword(),
                                ctx.link_targets().target())

        more_targets = ctx.additonal_targets()
        if more_targets:
            self.add_linked_targets(
                node,  more_targets.keyword(), more_targets.link_targets().target())

    def exitAdd_alias(self, ctx):
        alias = ctx.target(0).getText()
        target = ctx.target(1).getText()
        original: TargetNode = self.targets.get(target)
        if original is None:
            original = TargetNode(target)
            self.targets[target] = original

        self.targets[alias] = TargetNode(name=alias, alias_for=original)

    def exitAdd_interface(self, ctx):
        target = ctx.target().getText()
        self.targets[target] = TargetNode(name=target, is_interface=True)
