"""
AIME answer extraction and comparison utilities for lm-eval-harness.

Ported from evalscope (Qwen2.5-Math) to align scoring with evalscope's approach:
- Cascading answer extraction: \\boxed{} → "answer is" → last-number fallback
- Math-aware comparison: numeric equality, sympy symbolic equality
- Robust LaTeX normalization

Dependencies (all available in qwen_quant env):
  sympy, regex, latex2sympy2_extended, word2number
"""

import re
from typing import Dict, List

import regex as regex_mod
from latex2sympy2_extended import latex2sympy
from math import isclose
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from word2number import w2n


# ---------------------------------------------------------------------------
# process_results — lm-eval entry point (called via !function utils.process_results)
# ---------------------------------------------------------------------------

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    response = results[0]

    # Extract predicted answer using cascading fallback
    pred = extract_answer(response)

    # Get ground truth (handle both "Answer" and "answer" field names)
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])

    # Compare using math-aware equality
    correct = math_equal(pred, target)
    return {"exact_match": int(correct)}


# ---------------------------------------------------------------------------
# Answer extraction — cascading fallback from evalscope
# ---------------------------------------------------------------------------

def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    """Extract the answer from model output using cascading strategies.

    Priority order:
    1. minerva_math format ("final answer is $...$. I hope")
    2. \\boxed{...} with proper brace matching
    3. "he answer is" / "final answer is" (catches "The answer is" and "the answer is")
    4. Chinese: "答案是"
    5. "ANSWER:"
    6. Last number in the response (fallback)
    """
    pred_str = pred_str.replace('\u043a\u0438', '')

    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = tmp.split('$. I hope', 1)[0].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ''
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            pred = a
        else:
            a = ans.split('$')[0].strip()
            pred = a
    elif 'he answer is' in pred_str:
        pred = pred_str.split('he answer is')[-1].strip()
    elif 'final answer is' in pred_str:
        pred = pred_str.split('final answer is')[-1].strip()
    elif '答案是' in pred_str:
        pred = pred_str.split('答案是')[1].strip().split('\n\n')[0].strip()
    elif 'ANSWER:' in pred_str:
        pred = pred_str.split('ANSWER:')[-1].strip()
    else:
        if use_last_number:
            pattern = r'-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str.replace(',', ''))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
        else:
            pred = ''

    # Clean up
    pred = re.sub(r'\n\s*', '', pred)
    if pred != '' and pred[0] == ':':
        pred = pred[1:]
    if pred != '' and pred[-1] == '.':
        pred = pred[:-1]
    if pred != '' and pred[-1] == '/':
        pred = pred[:-1]
    pred = strip_answer_string(pred)
    return pred


# ---------------------------------------------------------------------------
# String normalization — from evalscope's strip_answer_string
# ---------------------------------------------------------------------------

def _convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        new_str += '{' + a + '}{' + b + '}' + substr[2:]
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        new_str += '{' + a + '}' + b + substr[2:]
                    else:
                        new_str += '{' + a + '}' + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        if 'sqrt' not in a:
            a = int(a)
        if 'sqrt' not in b:
            b = int(b)
        assert string == '{}/{}'.format(a, b)
        return '\\frac{' + str(a) + '}{' + str(b) + '}'
    except Exception:
        return string


def _fix_sqrt(string):
    return re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)


def strip_answer_string(string: str) -> str:
    """Normalize a math answer string for comparison."""
    string = str(string).strip()
    string = string.replace('\n', '')
    string = string.rstrip('.')

    # Remove inverse spaces
    string = string.replace('\\!', '')

    # Matrix normalization
    string = re.sub(r'\\begin\{array\}\{.*?\}', r'\\begin{pmatrix}', string)
    string = re.sub(r'\\end\{array\}', r'\\end{pmatrix}', string)
    string = string.replace('bmatrix', 'pmatrix')

    # Frac variants
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('\\neq', '\\ne').replace('\\leq', '\\le').replace('\\geq', '\\ge')

    # Remove delimiters
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('\\{', '{')
    string = string.replace('\\}', '}')

    # Convert word numbers in \text{...}
    def replace_match(match):
        word = match.group(1).lower()
        converted = _convert_word_number(word)
        return converted if converted != word else match.group(0)
    string = re.sub(r'\\text\{([a-zA-Z]+)\}', replace_match, string)

    # Remove squared units
    string = re.sub(r'(cm|inches)\}\^2', r'\1}', string)

    # Remove trailing \text{...} units
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string

    # Remove degrees
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # Remove dollar signs and parens
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = string.replace('\\(', '').replace('\\)', '')

    # Convert word numbers
    string = _convert_word_number(string)

    # Replace \text{...} → ...
    string = re.sub(r'\\text\{(.*?)\}', r'\1', string)
    for key in ['x=', 'y=', 'z=', 'x\\in', 'y\\in', 'z\\in', 'x\\to', 'y\\to', 'z\\to']:
        string = string.replace(key, '')
    string = string.replace('\\emptyset', r'{}')
    string = string.replace('(-\\infty,\\infty)', '\\mathbb{R}')

    # Remove percentage
    string = string.replace('\\%', '')
    string = string.replace(r'\%', '')
    string = string.replace('%', '')

    # Fix leading decimals
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # Strip wrapping brackets if alphanumeric content
    if (
        string.startswith('{') and string.endswith('}') and string.isalnum()
        or string.startswith('(') and string.endswith(')') and string.isalnum()
        or string.startswith('[') and string.endswith(']') and string.isalnum()
    ):
        string = string[1:-1]

    # Infinity
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    # Misc cleanup
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')
    string = re.sub(r'\\mbox{.*?}', '', string)
    string = string.replace("'", '')
    string = string.replace('"', '')

    # j → i (complex numbers)
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')

    # Remove trailing zeros: 7.000 → 7
    string = re.sub(r'(\d+)\.0*([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0*$', r'\1', string)

    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # Strip "k = " prefix
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # Remove unnecessary backslashes before integers
    string = re.sub(r'\\(?=\-?\d+(\\|\)|,|\]|$))', '', string)

    # Remove grade level suffix
    string = re.sub(r'thgrade$', '', string)

    # Normalize thousands-formatted numbers (e.g., 70,000 → 70000)
    if re.fullmatch(r'\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*', string):
        string = string.replace(',', '')

    # Sort comma-separated integer lists
    if re.fullmatch(r'(\s*-?\d+\s*,)*\s*-?\d+\s*', string):
        try:
            integer_list = sorted(map(int, string.split(',')))
            string = ','.join(map(str, integer_list))
        except Exception:
            pass

    return string


# ---------------------------------------------------------------------------
# Math comparison — from evalscope's math_equal + symbolic_equal
# ---------------------------------------------------------------------------

def _parse_digits(num):
    num = regex_mod.sub(',', '', str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith('%'):
            num = num[:-1]
            if num.endswith('\\'):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def _is_digit(num):
    return _parse_digits(num) is not None


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r'\{.*,.*\}', input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip('{}')
        pmatrix = r'\begin{pmatrix}' + m.replace(',', '\\') + r'\end{pmatrix}'
        pmatrix_list.append(pmatrix)
    return ', '.join(pmatrix_list)


def _symbolic_equal(a, b):
    """Check symbolic equality using multiple parsers and sympy simplification."""
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace('\\\\', '\\'))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # Direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # Simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # Equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    # Numeric equal via sympy N()
    try:
        if _numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # Matrix equal
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def math_equal(
    prediction,
    reference,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """Math-aware equality check.

    Returns True if prediction matches reference via:
    1. Exact string match (case-insensitive)
    2. Numeric equality (with optional percentage tolerance)
    3. Symbolic equality via sympy
    """
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    # 1. Numeric equality
    try:
        if _is_digit(prediction) and _is_digit(reference):
            pred_val = _parse_digits(prediction)
            ref_val = _parse_digits(reference)
            if include_percentage:
                gt_results = [ref_val / 100, ref_val, ref_val * 100]
            else:
                gt_results = [ref_val]
            for item in gt_results:
                try:
                    if is_close:
                        if _numeric_equal(pred_val, item):
                            return True
                    else:
                        if item == pred_val:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. Symbolic equality
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix handling
    if 'pmatrix' in prediction and 'pmatrix' not in reference:
        reference = _str_to_pmatrix(reference)

    # Strip brackets for comparison
    pred_str, ref_str = prediction, reference
    if (prediction.startswith('[') and prediction.endswith(']') and not reference.startswith('(')
        ) or (prediction.startswith('(') and prediction.endswith(')') and not reference.startswith('[')):
        pred_str = pred_str.strip('[]()')
        ref_str = ref_str.strip('[]()')
    for s in ['{', '}', '(', ')']:
        ref_str = ref_str.replace(s, '')
        pred_str = pred_str.replace(s, '')
    if pred_str.lower() == ref_str.lower():
        return True

    # List/tuple comparison: [a,b] vs [c,d]
    if (
        regex_mod.match(r'(\(|\[).+(\)|\])', prediction) is not None
        and regex_mod.match(r'(\(|\[).+(\)|\])', reference) is not None
    ):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix comparison
    if ((prediction.startswith('\\begin{pmatrix}') or prediction.startswith('\\begin{bmatrix}'))
        and (prediction.endswith('\\end{pmatrix}') or prediction.endswith('\\end{bmatrix}'))
        and (reference.startswith('\\begin{pmatrix}') or reference.startswith('\\begin{bmatrix}'))
        and (reference.endswith('\\end{pmatrix}') or reference.endswith('\\end{bmatrix}'))):
        pred_lines = [
            line.strip()
            for line in prediction[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split('&')
                ref_parts = ref_line.split('&')
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    # Equation comparison: a=b vs c=d → compare (a-b) vs (c-d)
    if prediction.count('=') == 1 and reference.count('=') == 1:
        pred = prediction.split('=')
        pred = f'{pred[0].strip()} - ({pred[1].strip()})'
        ref = reference.split('=')
        ref = f'{ref[0].strip()} - ({ref[1].strip()})'
        if _symbolic_equal(pred, ref) or _symbolic_equal(f'-({pred})', ref):
            return True
    elif (prediction.count('=') == 1 and len(prediction.split('=')[0].strip()) <= 2 and '=' not in reference):
        if math_equal(prediction.split('=')[1], reference, include_percentage, is_close):
            return True
    elif (reference.count('=') == 1 and len(reference.split('=')[0].strip()) <= 2 and '=' not in prediction):
        if math_equal(prediction, reference.split('=')[1], include_percentage, is_close):
            return True

    if _symbolic_equal(prediction, reference):
        return True

    return False
