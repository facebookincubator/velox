from antlr4.error.ErrorListener import ErrorListener
from antlr4.TokenStreamRewriter import TokenStreamRewriter
from antlr4 import CommonTokenStream
from antlr4.error.Errors import CancellationException
from .parser.CMakeListener import CMakeListener
from .parser.CMakeParser import CMakeParser
from . import io
import os
import re
from glob import glob


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
        if not name:
            raise Exception("Can not create target without name!")
        self.name: str = name
        self.headers: list[str] = list_if_none(headers)
        self.cpp_includes: list[str] = []
        self.h_includes: list[str] = []
        self.sources: list[str] = list_if_none(sources)
        # targets parsed from source files
        self.public_targets: list[TargetNode] = []
        self.private_targets: list[TargetNode] = []
        # targets parsed from cml
        self.ppublic_targets: list[TargetNode] = []
        self.pprivate_targets: list[TargetNode] = []
        self.is_interface = is_interface
        self.alias_for: TargetNode = alias_for
        self.cml_path: str = cml_path
        self.is_object_lib = False
        self.was_linked = False

    def __str__(self) -> str:
        message: str = self.name + ':\n'
        if self.alias_for:
            message += f"Alias for {self.alias_for.name}\n"
            return message + self.alias_for.__str__()

        if self.is_interface:
            message += 'Interface Target\n'

        message += 'Dir:\n'
        message += '' if self.cml_path is None else self.cml_path
        message += '\nSources:\n'
        message += '\n'.join(sorted(self.sources))
        message += '\nHeaders:\n'
        message += '\n'.join(sorted(self.headers))
        message += '\nPrivate includes:\n'
        message += '\n'.join(self.cpp_includes)
        message += '\nPublic includes:\n'
        message += '\n'.join(self.h_includes)
        message += '\nPublic Targets:\n'
        message += '\n'.join([t.name for t in self.public_targets])
        message += '\nPrivate Targets:'
        message += '\n'.join([t.name for t in self.private_targets])
        message += '\nPublic Targets(cml):\n'
        message += '\n'.join([t.name for t in self.ppublic_targets])
        message += '\nPrivate Targets(cml):'
        message += '\n'.join([t.name for t in self.private_targets])
        return message


class SyntaxErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise CancellationException(f"line {line}:{column} {msg}")


class TargetInputListener(CMakeListener):
    def __init__(self, target_dict: dict[str, TargetNode],
                 header_target_map=None, repo_root='') -> None:
        super().__init__()
        self.targets = target_dict
        self.repo_root = repo_root
        self.header_target_map = header_target_map

    def remove_cmake_var(self, items: list[str]) -> list[str]:
        cmake_var = re.compile(r'^\$\{.+\}$')
        return [i for i in items if not cmake_var.match(i)]

    def get_abs_path(self, files: list[str], ctx):

        # TODO add check that sources actually exist
        return files

    def add_linked_targets(self, node: TargetNode,
                           keyword: CMakeParser.KeywordContext,
                           targets: CMakeParser.TargetContext):
        target_names: list[str] = [t.getText() for t in targets]
        nodes: list[TargetNode] = []

        for target in target_names:
            target_node: TargetNode | None = self.targets.get(target)
            if target_node is None:
                target_node = TargetNode(target)
                self.targets[target] = target_node
            nodes.append(target_node)

        if keyword is None or keyword.public() or keyword.interface():
            node.ppublic_targets.extend(nodes)
        else:
            node.pprivate_targets.extend(nodes)

    def exitAdd_library(self, ctx: CMakeParser.Add_libraryContext):
        target = ctx.target().getText()

        files = None
        files = [file.getText() for file in ctx.source_file()]

        base_path = os.path.dirname(ctx.start.getInputStream().fileName)
        files = [f.replace('${CMAKE_CURRENT_LIST_DIR}/', base_path)
                 for f in files]
        files = self.remove_cmake_var(files)
        files = [os.path.join(base_path, f)
                 for f in files if not os.path.dirname(f)]

        sources = [f for f in files if os.path.splitext(
            f)[1] in [".c", ".cpp", ".cxx", ".cc", ".c++"]]
        headers = [h for h in glob(base_path + '/*.h*')
                   if io.has_matching_src(h, sources)]

        node = self.targets.get(target)
        if node is None:
            node = TargetNode(
                target, headers, sources, cml_path=base_path)
            self.targets[target] = node
        else:
            assert target == node.name
            assert node.sources == []
            node.sources = sources
            node.headers = headers
            node.cml_path = base_path

        lib_type = ctx.library_type()
        lib_type = '' if lib_type is None else lib_type.getText()
        if lib_type == 'OBJECT':
            node.is_object_lib = True

        if self.header_target_map is not None:
            for h in headers:
                self.header_target_map[h.removeprefix(self.repo_root)] = node

    def exitLink_libraries(self, ctx: CMakeParser.Link_librariesContext):
        # TODO handle executable targets
        target: str = ctx.target().getText()
        node: TargetNode | None = self.targets.get(target)
        if node is None:
            node = TargetNode(target)
            self.targets[target] = node

        # Skip any target_link_library calls after the primary one
        # as we have no way to model these properly
        if not node.was_linked:
            self.add_linked_targets(node, ctx.keyword(),
                                    ctx.link_targets().target())

            more_targets = ctx.additonal_targets()
            if more_targets:
                self.add_linked_targets(
                    node,  more_targets.keyword(),
                    more_targets.link_targets().target())

            node.was_linked = True

    def exitAdd_alias(self, ctx: CMakeParser.Add_aliasContext):
        alias = ctx.target(0).getText()
        target = ctx.target(1).getText()
        original: TargetNode | None = self.targets.get(target)
        if original is None:
            original = TargetNode(target)
            self.targets[target] = original

        self.targets[alias] = TargetNode(name=alias, alias_for=original)

    def exitAdd_interface(self, ctx):
        target = ctx.target().getText()
        self.targets[target] = TargetNode(name=target, is_interface=True)


class UpdateTargetsListener(CMakeListener):
    def __init__(self, targets: dict[str, TargetNode], token_stream: CommonTokenStream):
        super().__init__()
        self.token_stream = TokenStreamRewriter(token_stream)
        self.targets = targets

    def exitLink_libraries(self, ctx: CMakeParser.Link_librariesContext):
        target = self.targets.get(ctx.target().getText())
        assert target is not None

        def sort_targets(targets: list[str]):
            # We want to list the internal targets first
            a = [t for t in targets if t.startswith('velox')]
            b = [t for t in targets if not t.startswith('velox')]

            return sorted(a) + sorted(b)

        if not target.was_linked:
            public_targets = target.public_targets
            public_targets.extend(
                [t for t in target.ppublic_targets if t.is_object_lib
                 or t.is_interface or t.name.startswith('${')])
            private_targets = [
                t for t in target.private_targets if t not in public_targets]
            private_targets.extend(
                [t for t in target.pprivate_targets if t.is_object_lib
                 or t.is_interface or t.name.startswith('${')])
            public_targets = sort_targets(
                [*set([t.name for t in public_targets])])
            private_targets = sort_targets(
                [*set([t.name for t in private_targets])])
            start = ctx.start.tokenIndex + 2
            stop = ctx.stop.tokenIndex - 1
            p_text = f'PUBLIC {" ".join(public_targets)}' if public_targets else ''
            pr_text = f' PRIVATE {" ".join(private_targets)}' if private_targets else ''
            new = f'{target.name} ' + p_text + pr_text
            self.token_stream.replaceRange(start, stop, new)
            target.was_linked = True
        else:
            if ctx.keyword() is None:
                # if a target was linked with a keyword all other
                # occurences of target_link_libraries must also use
                # a keyword
                self.token_stream.insertAfter(
                    ctx.start.tokenIndex + 3, ' PUBLIC ')
