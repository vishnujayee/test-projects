"""Static safety check for agent-written code (docs/design.md 7.3 step 1).

Cheap and strict: parse with `ast`, allow only a small set of imports, reject anything
that could read files, run code, touch the network, or reach into the evaluator.
"""

import ast
from pathlib import Path

ALLOWED_MODULES = {
    "numpy",
    "math",
    "typing",
    "dataclasses",
    "trivec.quantizers.base",
    "trivec.quantizers.bits",
    "collections.abc",
}
# An agent's test file may also import pytest and the module it tests. Nothing else extra.
TEST_EXTRA_MODULES = frozenset({"pytest", "quantizer"})
FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",  # getattr(x, "__glo" + "bals__") would dodge the attribute check below
    "setattr",
    "delattr",
}
FORBIDDEN_ATTRS = {
    "load",
    "save",
    "savez",
    "savez_compressed",
    "fromfile",
    "tofile",
    "loadtxt",
    "savetxt",
    "genfromtxt",
    "memmap",
    "DataSource",
    "ctypeslib",
    "f2py",
    "distutils",
    "system",
    "popen",
    "__dict__",
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
    "__code__",
    "__closure__",
}
FORBIDDEN_NAMES = {"__builtins__", "__loader__", "__spec__"}
FORBIDDEN_MODULE_PREFIXES = (
    "trivec.eval",
    "trivec.lab",
    "trivec.data",
    "numpy.ctypeslib",
    "numpy.f2py",
    "numpy.distutils",
)


def _module_allowed(name: str, extra: frozenset[str]) -> bool:
    if name.startswith(FORBIDDEN_MODULE_PREFIXES):
        return False
    return name in ALLOWED_MODULES or name in extra or name.split(".")[0] in {"numpy", "math"}


def check_source(
    source: str, *, filename: str = "<agent>", extra_modules: frozenset[str] = frozenset()
) -> list[str]:
    """Return a list of violations; empty means the code may be run in the sandbox."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name, extra_modules):
                    problems.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or not _module_allowed(mod, extra_modules):
                problems.append(f"line {node.lineno}: import from {mod!r} is not allowed")
            for alias in node.names:
                if alias.name in FORBIDDEN_ATTRS or alias.name == "*":
                    problems.append(f"line {node.lineno}: importing {alias.name!r} is not allowed")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: call to {func.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"line {node.lineno}: attribute .{node.attr} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: name {node.id} is not allowed")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            problems.append(f"line {node.lineno}: global/nonlocal statements are not allowed")
    return problems


def check_file(path: Path) -> list[str]:
    return check_source(path.read_text(encoding="utf-8"), filename=str(path))


def check_test_file(path: Path) -> list[str]:
    """Same rules, but a test file may also import pytest and the quantizer it tests."""
    return check_source(
        path.read_text(encoding="utf-8"), filename=str(path), extra_modules=TEST_EXTRA_MODULES
    )


def has_required_contract(source: str) -> list[str]:
    """The module must define `build(**params)` and at least one class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["syntax error"]
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    problems = []
    if "build" not in names:
        problems.append("missing module-level function build(**params)")
    if not any(isinstance(n, ast.ClassDef) for n in tree.body):
        problems.append("missing a quantizer class")
    return problems
