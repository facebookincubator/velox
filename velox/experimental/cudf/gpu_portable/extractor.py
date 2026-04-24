#!/usr/bin/env python3
"""GPU-portable extractor: lift a simple Velox CPU function into a
self-contained __device__ function that cuDF's transform_extended can JIT.

Tree-sitter parses syntax only; it doesn't care about template instantiation
or name resolution, so every node (including nodes inside template-dependent
expressions like `std::is_integral_v<TNum>` or `std::round(x)`) is fully
structured.

Transforms (all AST/syntax-tree based):
  substitute_types   : `type_identifier` nodes whose text matches a binding.
  substitute_values  : `identifier` nodes whose text matches a binding.
  rewrite_returns    : `return_statement` nodes rewritten to out-param form.
  strip_std_cmath    : `call_expression` whose function is a
                       `qualified_identifier` starting with `std::` and in
                       a known cmath whitelist.
  pow_int_promotion  : `call_expression` named `pow` with first argument a
                       `number_literal` that's an integer.

Deps are pinned in requirements.txt alongside this file; the CMake build
provisions a venv for them under ${BINARY_DIR}/venv.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _load_ts():
    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser
    except ImportError as e:
        sys.exit(
            "gpu_portable extractor: tree-sitter bindings not importable.\n"
            "Expected a venv at ${VELOX_CUDF_GPU_PORTABLE_VENV} created by "
            "the CMake configure step; if you're invoking this script "
            "directly, install deps with:\n"
            "  python3 -m pip install -r "
            "velox/experimental/cudf/gpu_portable/requirements.txt\n"
            f"Underlying error: {e}"
        )
    return Language(tscpp.language()), Parser


def _parse(path: Path):
    lang, Parser = _load_ts()
    parser = Parser(lang)
    src = path.read_bytes()
    tree = parser.parse(src)
    return tree, src


def _find_function(root, name: str, src: bytes):
    """Find the `function_definition` whose declarator names `name`. Tries
    templated forms first (template_declaration wrapping function_definition),
    falls back to bare function_definition."""

    def declarator_name(function_def):
        # function_definition > function_declarator > identifier
        for child in function_def.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        return src[sub.start_byte : sub.end_byte].decode("utf-8")
        return None

    template_hit = None
    bare_hit = None

    def walk(node):
        nonlocal template_hit, bare_hit
        if node.type == "template_declaration":
            for child in node.children:
                if child.type == "function_definition":
                    if declarator_name(child) == name and template_hit is None:
                        template_hit = child
                        return
        if node.type == "function_definition":
            if declarator_name(node) == name and bare_hit is None:
                bare_hit = node
        for child in node.children:
            walk(child)

    walk(root)
    return template_hit or bare_hit


def _find_body(function_def):
    for child in function_def.children:
        if child.type == "compound_statement":
            return child
    return None


def _walk_preorder(node):
    yield node
    for child in node.children:
        yield from _walk_preorder(child)


# ---- Editor ----

@dataclass
class Edit:
    start: int
    end: int
    new_text: bytes
    origin: str = ""


@dataclass
class Editor:
    source: bytes
    edits: List[Edit] = field(default_factory=list)

    def replace(self, start, end, new_text, origin=""):
        if isinstance(new_text, str):
            new_text = new_text.encode("utf-8")
        self.edits.append(Edit(start, end, new_text, origin))

    def replace_node(self, node, new_text, origin=""):
        self.replace(node.start_byte, node.end_byte, new_text, origin)

    def apply(self):
        sorted_edits = sorted(self.edits, key=lambda e: e.start)
        for a, b in zip(sorted_edits, sorted_edits[1:]):
            if b.start < a.end:
                raise SystemExit(
                    f"Overlapping edits: [{a.origin}] {[a.start, a.end]} and "
                    f"[{b.origin}] {[b.start, b.end]}."
                )
        out = self.source
        for e in sorted(self.edits, key=lambda x: x.start, reverse=True):
            out = out[: e.start] + e.new_text + out[e.end :]
        return out

    def delta_within(self, start, end):
        return sum(
            len(e.new_text) - (e.end - e.start)
            for e in self.edits
            if start <= e.start < end
        )


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8")


# ---- Transforms ----

_STD_CMATH_NAMES = {
    "isfinite", "isnan", "isinf", "round", "trunc", "ceil", "floor",
    "pow", "fabs", "abs", "sqrt", "sin", "cos", "tan", "exp",
    "log", "log2", "log10",
}


def xform_substitute_types(body, editor, src, bindings):
    """Replace every `type_identifier` node whose text matches a binding."""
    if not bindings:
        return
    for node in _walk_preorder(body):
        if node.type == "type_identifier":
            text = _node_text(node, src)
            if text in bindings:
                editor.replace_node(node, bindings[text], origin="substitute_types")


def xform_substitute_values(body, editor, src, bindings):
    """Replace `identifier` nodes whose text matches a binding. Skips
    identifiers used as function names in a call_expression and function
    declarators, since those are never our target non-type template params."""
    if not bindings:
        return
    # Collect the ids we want to skip: any identifier that IS a function
    # being called (call_expression's `function` field).
    skip = set()
    for node in _walk_preorder(body):
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                skip.add((fn.start_byte, fn.end_byte))

    for node in _walk_preorder(body):
        if node.type == "identifier":
            if (node.start_byte, node.end_byte) in skip:
                continue
            text = _node_text(node, src)
            if text in bindings:
                editor.replace_node(node, bindings[text], origin="substitute_values")


def xform_rewrite_returns(body, editor, src, out_name):
    """Rewrite `return <expr>;` to `{ *<out_name> = <expr>; return; }`.
    Emitted as TWO surgical edits (prefix + suffix) rather than one
    whole-node replacement, so other transforms can still edit inside the
    expression without overlap errors."""
    if not out_name:
        return
    for node in _walk_preorder(body):
        if node.type != "return_statement":
            continue
        return_kw, semicolon, expr = None, None, None
        for c in node.children:
            if c.type == "return":
                return_kw = c
            elif c.type == ";":
                semicolon = c
            else:
                expr = c
        if expr is None:
            continue  # bare `return;`
        # Replace the `return` keyword with `{ *<out> =`.
        editor.replace(
            return_kw.start_byte, return_kw.end_byte,
            f"{{ *{out_name} =", origin="rewrite_returns_prefix",
        )
        # Insert ` return; }` after the `;` (or after the expr if no `;`).
        after = semicolon.end_byte if semicolon is not None else expr.end_byte
        trailing = " return; }" if semicolon is not None else "; return; }"
        editor.replace(after, after, trailing, origin="rewrite_returns_suffix")


def xform_strip_std_cmath(body, editor, src):
    """Drop the `std::` qualifier from call_expressions whose callee is a
    qualified_identifier `std::<cmath-fn>`."""
    for node in _walk_preorder(body):
        if node.type != "call_expression":
            continue
        fn = node.child_by_field_name("function")
        if fn is None or fn.type != "qualified_identifier":
            continue
        fn_text = _node_text(fn, src)
        if not fn_text.startswith("std::"):
            continue
        suffix = fn_text[len("std::"):]
        if suffix not in _STD_CMATH_NAMES:
            continue
        editor.replace_node(fn, suffix, origin="strip_std_cmath")


def xform_pow_int_promotion(body, editor, src):
    """Promote `pow(<int-literal>, ...)` first arg to double by appending
    `.0`. Catches both `pow(10, d)` and `std::pow(10, d)` after the cmath
    strip runs (but order-independent because we only match number_literal
    args that don't already contain a `.`)."""
    for node in _walk_preorder(body):
        if node.type != "call_expression":
            continue
        fn = node.child_by_field_name("function")
        if fn is None:
            continue
        fn_text = _node_text(fn, src)
        name = fn_text.split("::")[-1]
        if name != "pow":
            continue
        args = node.child_by_field_name("arguments")
        if args is None:
            continue
        # argument_list: `(` arg (, arg)* `)`
        first_arg = None
        for c in args.children:
            if c.type not in ("(", ",", ")"):
                first_arg = c
                break
        if first_arg is None or first_arg.type != "number_literal":
            continue
        text = _node_text(first_arg, src)
        if "." in text or "e" in text or "E" in text:
            continue
        editor.replace_node(first_arg, text + ".0", origin="pow_int_promotion")


# ---- Stages ----

WRAP_TEMPLATE = """\
// AUTO-GENERATED. Stage: wrap.
// Self-contained __device__ function inlined from a CPU function body.
// All transforms applied via syntax-tree nodes.

__device__ void {wrapper_name}({wrapper_signature}) {{
{body}
}}
"""


def cmd_locate(args):
    tree, src = _parse(args.input)
    node = _find_function(tree.root_node, args.func, src)
    if node is None:
        sys.exit(f"Not found: {args.func}")
    print(f"found function_definition '{args.func}'")
    print(f"bytes: [{node.start_byte}, {node.end_byte})")
    print(f"lines: {node.start_point[0] + 1}:{node.start_point[1] + 1} -> "
          f"{node.end_point[0] + 1}:{node.end_point[1] + 1}")


def cmd_extract(args):
    tree, src = _parse(args.input)
    node = _find_function(tree.root_node, args.func, src)
    if node is None:
        sys.exit(f"Not found: {args.func}")
    args.output.write_bytes(src[node.start_byte : node.end_byte])
    print(f"Wrote {args.output}", file=sys.stderr)


def _parse_param(text):
    """Split 'TYPE NAME' (possibly with `*`, `&`, or `::`) into (type_str,
    name). The name is the last whitespace-separated token; everything
    before it is the type."""
    tokens = text.strip().rsplit(None, 1)
    if len(tokens) != 2 or not tokens[1]:
        sys.exit(f"Expected 'TYPE NAME', got: {text!r}")
    return tokens[0], tokens[1]


def _bake_placeholder(name):
    """Internal token spliced into the wrapped source at the bridge-line
    site, replaced with std::to_string(<name>) by the embed stage. Never
    user-visible."""
    return f"__GPP_BAKE_{name.upper()}__"


def cmd_wrap(args):
    template_types = dict(kv.split("=", 1) for kv in (args.template_type or []))
    template_values = dict(kv.split("=", 1) for kv in (args.template_value or []))

    out_type, out_name = _parse_param(args.out_param)
    input_params = [_parse_param(p) for p in (args.input_param or [])]
    baked_params = [_parse_param(p) for p in (args.baked_param or [])]

    # Wrapper signature is out + inputs. Baked params are compile-time
    # literals spliced into the body at JIT time; they are not wrapper
    # parameters.
    sig_parts = [args.out_param.strip()]
    sig_parts.extend(p.strip() for p in (args.input_param or []))
    wrapper_signature = ", ".join(sig_parts)

    tree, src = _parse(args.input)
    func = _find_function(tree.root_node, args.func, src)
    if func is None:
        sys.exit(f"Function '{args.func}' not found in {args.input}.")
    body = _find_body(func)
    if body is None:
        sys.exit(f"Function '{args.func}' has no body.")

    editor = Editor(source=src)

    # Order: substitute first so the return rewriter (which captures the
    # expression text) sees concrete types; strip_std_cmath before
    # pow_int_promotion so the pow matcher can rely on `pow` as the
    # unqualified function name.
    xform_substitute_types(body, editor, src, template_types)
    xform_substitute_values(body, editor, src, template_values)
    xform_rewrite_returns(body, editor, src, out_name)
    xform_strip_std_cmath(body, editor, src)
    xform_pow_int_promotion(body, editor, src)

    edited = editor.apply()

    new_body_end = body.end_byte + editor.delta_within(body.start_byte, body.end_byte)
    body_text = edited[body.start_byte + 1 : new_body_end - 1].decode("utf-8")

    # Synthesize the bridge lines for baked params: each declares a local
    # of the same name and type used inside the CPU body, initialized from
    # a unique placeholder token that the embed stage splices out.
    bridge_lines = [
        f"const {type_} {name} = {_bake_placeholder(name)};"
        for type_, name in baked_params
    ]
    prelude = ("\n".join(bridge_lines) + "\n") if bridge_lines else ""
    body_out = prelude + body_text.strip()
    indented = "\n".join(
        ("  " + line) if line.strip() else line for line in body_out.splitlines()
    )

    out = WRAP_TEMPLATE.format(
        wrapper_name=args.wrapper_name,
        wrapper_signature=wrapper_signature,
        body=indented,
    )
    args.output.write_text(out)
    print(f"Wrote {args.output} ({len(out)} bytes).", file=sys.stderr)


# embed stage: wraps the __device__ source string in a header that exposes
# it to host code. Baked params (values known at call time, e.g. a SQL
# literal scale) get their placeholder splinced out and replaced with
# std::to_string(<name>) so NVRTC sees them as literal constants at JIT
# time. With no baked params, a parameterless getter is emitted.
EMBED_HEADER = """\
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// AUTO-GENERATED. Stage: embed.
// Source: {source_path}

#pragma once

#include <string>

namespace facebook::velox::cudf_velox::gpu_portable {{
"""

EMBED_FOOTER = "\n} // namespace facebook::velox::cudf_velox::gpu_portable\n"


def _build_embed_body(source, bakes, wrapper_name):
    """Build the getter function body. With no bakes, returns a parameterless
    function returning the raw source. With bakes, splices std::to_string(var)
    at each placeholder occurrence."""
    if not bakes:
        return (
            f"inline std::string {wrapper_name}_source() {{\n"
            f'  return R"__VELOX_GPU__({source})__VELOX_GPU__";\n'
            f"}}\n"
        )

    # Collect occurrences of every placeholder, in document order.
    positions = []  # (start, end, var_name)
    for placeholder, _, var_name in bakes:
        start = 0
        found = False
        while True:
            idx = source.find(placeholder, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(placeholder), var_name))
            start = idx + len(placeholder)
            found = True
        if not found:
            sys.exit(f"Placeholder {placeholder!r} not found in wrapped source.")

    positions.sort()
    for a, b in zip(positions, positions[1:]):
        if b[0] < a[1]:
            sys.exit("Placeholder occurrences overlap in wrapped source.")

    sig = ", ".join(param_decl for _, param_decl, _ in bakes)
    lines = [f"inline std::string {wrapper_name}_source({sig}) {{"]

    # One std::to_string per unique parameter, names like `<var>Str`.
    seen = []
    for _, _, var_name in bakes:
        if var_name not in seen:
            lines.append(
                f"  const std::string {var_name}Str = std::to_string({var_name});"
            )
            seen.append(var_name)

    cursor = 0
    first = True
    for pos_start, pos_end, var_name in positions:
        chunk = source[cursor:pos_start]
        op = "std::string src =" if first else "src +="
        lines.append(f'  {op} R"__VELOX_GPU__({chunk})__VELOX_GPU__";')
        lines.append(f"  src += {var_name}Str;")
        cursor = pos_end
        first = False
    trailing = source[cursor:]
    lines.append(f'  src += R"__VELOX_GPU__({trailing})__VELOX_GPU__";')
    lines.append("  return src;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def cmd_embed(args):
    source = args.source.read_text()
    baked_params = [_parse_param(p) for p in (args.baked_param or [])]
    bakes = [
        (_bake_placeholder(name), f"{type_} {name}", name)
        for type_, name in baked_params
    ]
    body_code = _build_embed_body(source, bakes, args.wrapper_name)
    out = (
        EMBED_HEADER.format(source_path=args.source.as_posix())
        + "\n"
        + body_code
        + EMBED_FOOTER
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out)
    print(f"Wrote {args.output}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    loc = sub.add_parser("locate")
    loc.add_argument("--input", required=True, type=Path)
    loc.add_argument("--func", required=True)
    loc.set_defaults(func_=cmd_locate)

    ext = sub.add_parser("extract")
    ext.add_argument("--input", required=True, type=Path)
    ext.add_argument("--func", required=True)
    ext.add_argument("--output", required=True, type=Path)
    ext.set_defaults(func_=cmd_extract)

    wrp = sub.add_parser("wrap")
    wrp.add_argument("--input", required=True, type=Path)
    wrp.add_argument("--func", required=True)
    wrp.add_argument("--wrapper-name", required=True)
    wrp.add_argument(
        "--template-type",
        action="append",
        metavar="TEMPLATE_TYPE_PARAM=CONCRETE_TYPE",
        help=(
            "Pick the concrete type for a template type parameter of the "
            "CPU function being lifted (e.g. `TNum=double`). Repeatable."
        ),
    )
    wrp.add_argument(
        "--template-value",
        action="append",
        metavar="TEMPLATE_NONTYPE_PARAM=LITERAL",
        help=(
            "Pick the concrete value for a template non-type parameter of "
            "the CPU function (e.g. `alwaysRoundNegDec=false`). Repeatable."
        ),
    )
    wrp.add_argument(
        "--out-param",
        required=True,
        metavar="TYPE NAME",
        help=(
            "The out-pointer parameter of the emitted __device__ wrapper "
            "(e.g. `double* out`). Return statements in the CPU body are "
            "rewritten to `*NAME = expr; return;`."
        ),
    )
    wrp.add_argument(
        "--input-param",
        action="append",
        metavar="TYPE NAME",
        help=(
            "A per-row input parameter of the emitted __device__ wrapper "
            "(e.g. `double number`). Repeatable."
        ),
    )
    wrp.add_argument(
        "--baked-param",
        action="append",
        metavar="TYPE NAME",
        help=(
            "A value known at call time, spliced into the emitted source "
            "as a literal so NVRTC can constant-fold. The embed stage "
            "takes TYPE NAME on the getter and substitutes "
            "std::to_string(NAME) at JIT time. Repeatable."
        ),
    )
    wrp.add_argument("--output", required=True, type=Path)
    wrp.set_defaults(func_=cmd_wrap)

    emb = sub.add_parser("embed")
    emb.add_argument("--source", required=True, type=Path)
    emb.add_argument("--wrapper-name", required=True)
    emb.add_argument(
        "--baked-param",
        action="append",
        metavar="TYPE NAME",
        help=(
            "Must match the --baked-param values passed to `wrap`. "
            "Repeatable. Omit for a parameterless getter."
        ),
    )
    emb.add_argument("--output", required=True, type=Path)
    emb.set_defaults(func_=cmd_embed)

    args = p.parse_args()
    args.func_(args)


if __name__ == "__main__":
    main()
