"""Load one provider against the real upstream interface without every vendor SDK.

The full evaluator imports all provider SDKs unconditionally. This smoke/test
loader executes only imports, classes and functions from providers/__init__.py,
omitting its final auto-import block. The normal upstream run uses that block.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import types


PROVIDERS = Path(__file__).resolve().parents[2] / "providers"


def load_provider():
    interface = PROVIDERS / "__init__.py"
    source = interface.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(interface))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level == 1)]
    package_name = "speech_expert_eval_providers"
    package = types.ModuleType(package_name)
    package.__path__ = [str(interface.parent)]
    sys.modules[package_name] = package
    exec(compile(tree, str(interface), "exec"), package.__dict__)
    module_name = f"{package_name}.speech_expert_provider"
    spec = importlib.util.spec_from_file_location(module_name, PROVIDERS / "speech_expert_provider.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, package
