import ast
import pathlib

from qrc_vs_qelm import core


def _public_defs(module) -> set[str]:
    path = pathlib.Path(module.__file__)
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.add(target.id)
    return names


def test_core_all_exports_public_submodule_definitions():
    exported = set(core.__all__)
    modules = [
        core.diagnostics,
        core.dissipators,
        core.liouvillian,
        core.operators,
        core.readout,
        core.reservoirs,
        core.sparse_evolve,
        core.tasks,
    ]
    missing = {
        module.__name__.rsplit(".", 1)[-1]: sorted(_public_defs(module) - exported)
        for module in modules
        if _public_defs(module) - exported
    }
    assert missing == {}
