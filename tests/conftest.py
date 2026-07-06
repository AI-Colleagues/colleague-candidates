"""Shared helpers for testing self-contained colleague workflow modules."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


COLLEAGUES_DIR = Path(__file__).resolve().parent.parent / "colleagues"


def load_workflow_module(relative_path: str) -> ModuleType:
    """Import a colleague's workflow.py as an isolated module.

    Every colleague directory ships its own ``workflow.py`` with no package
    ``__init__.py`` (workflows are intentionally self-contained), so a plain
    ``import workflow`` would collide across colleagues in ``sys.modules``.
    Loading each file under a name derived from its path keeps them isolated.
    """
    path = COLLEAGUES_DIR / relative_path / "workflow.py"
    module_name = "colleague_workflow_" + relative_path.replace("/", "_")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
