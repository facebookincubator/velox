from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import CancellationException
from .parser.CMakeListener import CMakeListener
from .parser.CMakeParser import CMakeParser
import os
def list_if_none(arg: list | None) -> list:
    if arg is None:
        return []
    else:
        return arg


class TargetNode:
    def __init__(
            self, name: str, headers: list[str] = None, sources: list[str] = None,
            is_interface: bool = False, alias_for=None, cml_path=None
    ) -> None:
        self.name: str = name
        self.headers: list[str] = list_if_none(headers)
        self.public_headers: list[str] = []
        self.sources: list[str] = list_if_none(sources)
        self.public_targets: list[TargetNode] = []
        self.private_targets: list[TargetNode] = []
        self.is_interface = is_interface
        self.alias_for: TargetNode = alias_for
        self.cml_path: str = cml_path

    def __str__(self) -> str:
        message: str = self.name + ':\n'
        if self.alias_for:
            message += f"Alias for {self.alias_for.name}\n"
            return message + self.alias_for.__str__()

        if self.is_interface:
            message += 'Interface Target\n'

        message += 'Sources:\n'
        message += '\n'.join(self.sources)
        message += '\nPublic Targets:\n'
        message += '\n'.join([t.name for t in self.public_targets])
        message += '\nPrivate Targets:'
        message += '\n'.join([t.name for t in self.private_targets])
        return message


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
