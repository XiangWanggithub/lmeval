"""LiveCodeBench test execution engine.

Ported from evalscope's live_code_bench/testing_util.py with minimal changes:
- Removed evalscope logger dependency (use stdlib logging)
- Self-contained — no evalscope imports

Supports two execution modes:
- call_based: LeetCode-style (import function, call with args, compare return value)
- standard_input: Competitive programming (patch stdin, capture stdout, compare lines)

Safety: reliability_guard() disables destructive OS functions. This only runs
inside a subprocess spawned by multiprocessing, never in the main process.
"""

import ast
import json
import platform
import signal
import sys
import time
from decimal import Decimal
from enum import Enum
from functools import partial
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

import_string = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\n"
    "import heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)


def truncatefn(s, length=300):
    s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1])
                    + "\n"
                    + ast.unparse(last_block.body)
                )
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)
        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        # Execute generated code in an isolated module namespace
        exec(code, tmp_sol.__dict__)  # noqa: S102 — runs in sandboxed subprocess only
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str):
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code, all_inputs, all_outputs, fn_name, timeout):
    import numpy as np

    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [-2], {"error_code": -2, "error_message": "Compilation failed"}

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return [-2], {"error_code": -2, "error_message": f"Function {fn_name} not found"}

    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
        finally:
            signal.alarm(0)

    return all_results, {"execution time": total_execution}


def grade_stdio(code, all_inputs, all_outputs, timeout):
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [-2], {"error_code": -2, "error_message": "Compilation failed"}

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return [-2], {"error_code": -2, "error_message": "wrapped_function not found"}

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
            finally:
                signal.alarm(0)

        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        WA_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_args

        for line_idx, (pred_line, gt_line) in enumerate(
            zip(stripped_prediction_lines, stripped_gt_out_lines)
        ):
            if pred_line == gt_line:
                continue
            success, decimal_pred = convert_line_to_decimals(pred_line)
            if not success:
                all_results.append(-2)
                WA_args["error_message"] = f"Wrong answer at line {line_idx}"
                return all_results, WA_args
            success, decimal_gt = convert_line_to_decimals(gt_line)
            if not success:
                all_results.append(-2)
                WA_args["error_message"] = f"Wrong answer at line {line_idx}"
                return all_results, WA_args
            if decimal_pred == decimal_gt:
                continue
            all_results.append(-2)
            WA_args["error_message"] = f"Wrong answer at line {line_idx}"
            return all_results, WA_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    """Run test cases against generated code in a sandboxed subprocess."""
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e

    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based
        method_name = in_outs["fn_name"]

    if test is None:
        return [-1], {"error": "no test code provided"}

    if which_type == CODE_TYPE.call_based:
        signal.alarm(timeout)
        try:
            results, metadata = grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            return [-4], {"error_code": -4, "error_message": f"Error: {e}"}
        finally:
            signal.alarm(0)
    else:
        signal.alarm(timeout)
        try:
            results, metadata = grade_stdio(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            return [-4], {"error_code": -4, "error_message": f"Error: {e}"}
        finally:
            signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """Disable destructive functions in the subprocess."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    import builtins
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
