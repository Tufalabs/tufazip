#!/usr/bin/env python3
"""Build the ranked decoder ZIP (deflate level 9) and print its size.

Python sources lose comments, docstrings and blank lines on the way in. Each
stripped file must parse to the same AST as the original minus docstrings, so
the shipped code is semantically identical to the repository code.
"""
import ast
import io
import os
import sys
import tokenize
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = ("tufazip", "nn.py", "arithmetic_coder.py", "checkpoint_utils.py",
          "wunpack.py")
OTHER = ("detok.c", "weights.tz", "enwik9.voc", "pyproject.toml", "uv.lock",
         "README.md", "LICENSE", "NOTICE")


def _docstring_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (len(body) > 1 and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                yield node, body[0]


def _without_docstrings(tree):
    for node, doc in list(_docstring_nodes(tree)):
        node.body.remove(doc)
    return tree


def strip_python(source):
    lines = source.splitlines(keepends=True)
    drop, cut, protected = set(), {}, set()
    for _, doc in _docstring_nodes(ast.parse(source)):
        drop.update(range(doc.lineno, doc.end_lineno + 1))
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            if not (tok.start[0] == 1 and tok.string.startswith("#!")):
                cut[tok.start[0]] = tok.start[1]
        elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    out = []
    for number, line in enumerate(lines, 1):
        if number in drop:
            continue
        if number in cut:
            line = line[:cut[number]].rstrip() + "\n"
        if line.strip() or number in protected:
            out.append(line)
    stripped = "".join(out)
    expected = ast.dump(_without_docstrings(ast.parse(source)))
    if ast.dump(ast.parse(stripped)) != expected:
        raise SystemExit("AST changed while stripping")
    return stripped


def build(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in PYTHON + OTHER:
            data = open(os.path.join(ROOT, name), "rb").read()
            if name in PYTHON:
                data = strip_python(data.decode()).encode()
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if name == "tufazip" else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            packed = zipfile._get_compressor(zipfile.ZIP_DEFLATED, 9)
            if len(packed.compress(data) + packed.flush()) >= len(data):
                info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, data, compresslevel=9)
    return os.path.getsize(path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist", "decoder.zip")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    print("{}: {} bytes".format(target, build(target)))
