"""Curve fitting shared by Analyze ▸ Curve Fitting and Analyze ▸ Calibrate (ImageJ CurveFitter equivalents).

Besides the built-in functions, :func:`fit` accepts a **custom equation** (ImageJ's "*User-defined*" fit), e.g.
``y = a*exp(-b*x) + c``: the text is parsed with :mod:`ast` into a tree of whitelisted operators, functions and
names (parameters a–f, the variable x, pi) and evaluated by walking that tree – no ``eval`` of arbitrary code.
"""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from scipy.special import erf as _erf, erfc as _erfc

# name -> (formula, number of parameters, function(x, *p))
FIT_FUNCTIONS: dict[str, tuple[str, int, Callable[..., np.ndarray]]] = {}


def _reg(name: str, formula: str, n: int):
    def deco(fn):
        FIT_FUNCTIONS[name] = (formula, n, fn)
        return fn
    return deco


@_reg("Straight Line", "y = a + b·x", 2)
def _line(x, a, b):
    return a + b * x


@_reg("2nd Degree Polynomial", "y = a + b·x + c·x²", 3)
def _poly2(x, a, b, c):
    return a + b * x + c * x ** 2


@_reg("3rd Degree Polynomial", "y = a + b·x + c·x² + d·x³", 4)
def _poly3(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


@_reg("4th Degree Polynomial", "y = a + b·x + c·x² + d·x³ + e·x⁴", 5)
def _poly4(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


@_reg("5th Degree Polynomial", "y = a + b·x + c·x² + d·x³ + e·x⁴ + f·x⁵", 6)
def _poly5(x, *p):
    return np.polyval(p[::-1], x)


@_reg("6th Degree Polynomial", "y = a + b·x + … + g·x⁶", 7)
def _poly6(x, *p):
    return np.polyval(p[::-1], x)


@_reg("7th Degree Polynomial", "y = a + b·x + … + h·x⁷", 8)
def _poly7(x, *p):
    return np.polyval(p[::-1], x)


@_reg("8th Degree Polynomial", "y = a + b·x + … + i·x⁸", 9)
def _poly8(x, *p):
    return np.polyval(p[::-1], x)


@_reg("Exponential", "y = a·exp(b·x)", 2)
def _exp(x, a, b):
    return a * np.exp(b * x)


@_reg("Exponential (linear regression)", "y = a·exp(b·x)", 2)
def _exp_regression(x, a, b):
    return a * np.exp(b * x)


@_reg("Exponential with Offset", "y = a·exp(−b·x) + c", 3)
def _exp_off(x, a, b, c):
    return a * np.exp(-b * x) + c


@_reg("Exponential Recovery", "y = a·(1 − exp(−b·x)) + c", 3)
def _exp_rec(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c


@_reg("Power", "y = a·x^b", 2)
def _power(x, a, b):
    return a * np.power(np.abs(x), b)


@_reg("Power (linear regression)", "y = a·x^b", 2)
def _power_regression(x, a, b):
    return a * np.power(np.abs(x), b)


@_reg("Log", "y = a·ln(b·x)", 2)
def _log(x, a, b):
    return a * np.log(np.abs(b * x) + 1e-300)


@_reg("Log2", "y = a + b·ln(x − c)", 3)
def _log2(x, a, b, c):
    # ln(x − c) is only defined right of c; a tiny floor keeps the least-squares residuals finite (acts as a penalty)
    return a + b * np.log(np.maximum(x - c, 1e-300))


@_reg("Gaussian", "y = a + (b − a)·exp(−(x − c)² / (2d²))", 4)
def _gauss(x, a, b, c, d):
    return a + (b - a) * np.exp(-((x - c) ** 2) / (2 * d * d + 1e-300))


@_reg("Gaussian (no offset)", "y = a·exp(−(x − b)² / (2c²))", 3)
def _gauss_no_offset(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c * c + 1e-300))


@_reg("Sigmoid (Boltzmann)", "y = a + (b − a) / (1 + exp((c − x) / d))", 4)
def _sigmoid(x, a, b, c, d):
    return a + (b - a) / (1 + np.exp(np.clip((c - x) / (d if d != 0 else 1e-12), -500, 500)))


@_reg("Error Function", "y = a + b·erf((x − c) / d)", 4)
def _erf_fn(x, a, b, c, d):
    return a + b * _erf((x - c) / (d if d != 0 else 1e-12))


@_reg("Rodbard", "y = d + (a − d) / (1 + (x/c)^b)", 4)
def _rodbard(x, a, b, c, d):
    return d + (a - d) / (1 + np.power(np.abs(x / (c if c != 0 else 1e-12)), b))


@_reg("Rational", "y = (a + b·x) / (1 + c·x)", 3)
def _rational(x, a, b, c):
    return (a + b * x) / (1 + c * x)


@_reg("Michaelis-Menten", "y = a·x / (b + x)", 2)
def _mm(x, a, b):
    return a * x / (b + x)


@_reg("Gamma Variate", "y = b·(x − a)^c·exp(−(x − a)/d)", 4)
def _gamma_variate(x, a, b, c, d):
    dx = np.maximum(x - a, 0.0)
    with np.errstate(all="ignore"):
        return np.where(dx > 0, b * np.power(dx, c) * np.exp(-dx / (d if d != 0 else 1e-12)), 0.0)


@_reg("Chapman-Richards", "y = a·(1 − exp(−b·x))^c", 3)
def _chapman(x, a, b, c):
    with np.errstate(all="ignore"):
        return a * np.power(np.maximum(1.0 - np.exp(-b * x), 0.0), c)


def _inverse_rodbard(x, a, b, c, d):
    # the Rodbard function solved for its argument: y = c·((x − a) / (d − x))^(1/b)
    with np.errstate(all="ignore"):
        ratio = (x - a) / (d - x)
        return c * np.power(np.abs(ratio), 1.0 / (b if b != 0 else 1e-12))


# ImageJ RODBARD2 ('Rodbard (NIH Image)'): the Rodbard fit with x and y swapped, reported in the inverse form.
# ImageJ's INV_RODBARD ('Inverse Rodbard', the same formula minimised directly) is accepted as an alias of it –
# tests/test_analysis_more.py pins that spelling to 'Rodbard (NIH Image)'; register _inverse_rodbard under its own
# name here to expose the direct least-squares variant (initial guesses are already in place).
_reg("Rodbard (NIH Image)", "y = c·((x − a) / (d − x))^(1/b)", 4)(_inverse_rodbard)


FIT_NAMES = list(FIT_FUNCTIONS.keys())
CUSTOM = "Custom"
# ImageJ CurveFitter spellings accepted as aliases (case-insensitive lookup in fit())
FIT_ALIASES = {
    "exponential with offset": "Exponential with Offset", "exponential recovery": "Exponential Recovery",
    "gamma variate": "Gamma Variate", "chapman-richards": "Chapman-Richards", "chapman richards": "Chapman-Richards",
    "rodbard (nih image)": "Rodbard (NIH Image)", "rodbard2": "Rodbard (NIH Image)", "nih rodbard": "Rodbard (NIH Image)",
    "inverse rodbard": "Rodbard (NIH Image)", "inv rodbard": "Rodbard (NIH Image)", "inv_rodbard": "Rodbard (NIH Image)",
    "sigmoid": "Sigmoid (Boltzmann)", "boltzmann": "Sigmoid (Boltzmann)", "poly2": "2nd Degree Polynomial",
    "poly3": "3rd Degree Polynomial", "poly4": "4th Degree Polynomial", "poly5": "5th Degree Polynomial",
    "poly6": "6th Degree Polynomial", "poly7": "7th Degree Polynomial", "poly8": "8th Degree Polynomial", "linear": "Straight Line",
    "erf": "Error Function", "error function": "Error Function", "y = a+b*ln(x-c)": "Log2", "log2": "Log2",
    "gaussian (no offset)": "Gaussian (no offset)", "gaussian no offset": "Gaussian (no offset)", "gaussian_nooffset": "Gaussian (no offset)",
    "exponential (linear regression)": "Exponential (linear regression)", "exp_regression": "Exponential (linear regression)",
    "power (linear regression)": "Power (linear regression)", "power_regression": "Power (linear regression)",
    "custom": CUSTOM, "user-defined": CUSTOM, "*user-defined*": CUSTOM, "user defined": CUSTOM,
}


def canonical_name(name: str) -> str:
    if name in FIT_FUNCTIONS or name == CUSTOM:
        return name
    low = str(name).strip().lower()
    if low in FIT_ALIASES:
        return FIT_ALIASES[low]
    for k in FIT_FUNCTIONS:
        if k.lower() == low:
            return k
    return name


PARAM_NAMES = "abcdefghi"
CUSTOM_PARAM_NAMES = "abcdef"  # ImageJ's user-defined fits take up to six parameters a–f


# ---------------------------------------------------------------------------------------- custom equations
_EQ_PREFIX = re.compile(r"^\s*(?:y|f\s*\(\s*x\s*\))\s*=\s*", re.IGNORECASE)
_SAFE_FUNCTIONS: dict[str, tuple[Callable[..., Any], int]] = {  # name -> (numpy implementation, arity)
    "sin": (np.sin, 1), "cos": (np.cos, 1), "tan": (np.tan, 1), "asin": (np.arcsin, 1), "acos": (np.arccos, 1), "atan": (np.arctan, 1),
    "sinh": (np.sinh, 1), "cosh": (np.cosh, 1), "tanh": (np.tanh, 1), "exp": (np.exp, 1), "log": (np.log, 1), "ln": (np.log, 1),
    "log10": (np.log10, 1), "log2": (np.log2, 1), "sqrt": (np.sqrt, 1), "pow": (np.power, 2), "abs": (np.abs, 1),
    "erf": (_erf, 1), "erfc": (_erfc, 1), "floor": (np.floor, 1), "ceil": (np.ceil, 1), "round": (np.round, 1),
    "min": (np.minimum, 2), "max": (np.maximum, 2),
}
_SAFE_CONSTANTS = {"pi": math.pi}
_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
                                                     ast.Pow: np.power, ast.Mod: np.mod}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.USub: np.negative, ast.UAdd: lambda v: v}


class CustomEquation:
    """A user-defined fit function ``y = f(x; a, b, …)`` compiled from text into a validated :mod:`ast` tree.

    Allowed: numbers, the variable ``x``, parameters ``a``–``f``, ``pi``, the operators ``+ - * / ^ ** %``, unary ``-``,
    and the functions sin cos tan asin acos atan sinh cosh tanh exp log ln log10 log2 sqrt pow abs erf erfc floor ceil
    round min max.  Everything else (attributes, subscripts, calls of other names, comparisons, lambdas, …) is rejected.
    """

    def __init__(self, text: str) -> None:
        self.text = str(text).strip()
        expr = _EQ_PREFIX.sub("", self.text).strip()
        if not expr:
            raise ValueError("Custom equation is empty – e.g. y = a*exp(-b*x) + c")
        expr = expr.replace("^", "**")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Cannot parse equation {self.text!r}: {e.msg}") from None
        used: set[str] = set()
        self._validate(tree.body, used)
        self.params = [p for p in CUSTOM_PARAM_NAMES if p in used]
        if not self.params:
            raise ValueError("A custom equation needs at least one parameter (a–f), e.g. y = a + b*x")
        self.tree = tree.body
        self.n = len(self.params)
        self.formula = "y = " + expr.replace("**", "^")

    @classmethod
    def _validate(cls, node: ast.AST, used: set[str]) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Only numbers are allowed in a custom equation (got {node.value!r})")
        elif isinstance(node, ast.Name):
            if node.id in CUSTOM_PARAM_NAMES:
                used.add(node.id)
            elif node.id not in ("x",) and node.id not in _SAFE_CONSTANTS:
                raise ValueError(f"Unknown name {node.id!r} in custom equation: parameters are a–f, the variable is x")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY_OPS:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed in a custom equation")
            cls._validate(node.left, used)
            cls._validate(node.right, used)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed in a custom equation")
            cls._validate(node.operand, used)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError(f"Only these functions are allowed in a custom equation: {', '.join(sorted(_SAFE_FUNCTIONS))}")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in a custom equation")
            arity = _SAFE_FUNCTIONS[node.func.id][1]
            if len(node.args) != arity:
                raise ValueError(f"{node.func.id}() takes {arity} argument{'s' if arity != 1 else ''} in a custom equation")
            for a in node.args:
                cls._validate(a, used)
        else:
            raise ValueError(f"{type(node).__name__} is not allowed in a custom equation")

    def _eval(self, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return _SAFE_CONSTANTS[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY_OPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](self._eval(node.operand, env))
        if isinstance(node, ast.Call):
            fn = _SAFE_FUNCTIONS[node.func.id][0]  # type: ignore[union-attr]
            return fn(*[self._eval(a, env) for a in node.args])
        raise ValueError(type(node).__name__)  # pragma: no cover – rejected by _validate

    def __call__(self, x: Any, *p: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        env: dict[str, Any] = {"x": x}
        env.update(zip(self.params, p))
        with np.errstate(all="ignore"):
            out = self._eval(self.tree, env)
        return np.broadcast_to(np.asarray(out, dtype=np.float64), x.shape).copy()


@lru_cache(maxsize=64)
def compile_equation(text: str) -> CustomEquation:
    """Parse (and cache) a custom equation; raises ValueError for anything outside the whitelist."""
    return CustomEquation(text)


# ---------------------------------------------------------------------------------------- initial guesses
def _initial_guess(name: str, x: np.ndarray, y: np.ndarray) -> list[float]:
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    span = (xmax - xmin) or 1.0
    if name in ("Exponential", "Exponential (linear regression)"):
        pos = y > 0
        if pos.sum() >= 2:
            b, loga = np.polyfit(x[pos], np.log(y[pos]), 1)
            return [float(np.exp(loga)), float(b)]
        return [ymax or 1.0, -1.0 / span]
    if name == "Exponential with Offset":
        c = ymin
        return [float(y[0] - c) or 1.0, 1.0 / span, c]
    if name == "Exponential Recovery":
        return [float(ymax - y[0]) or 1.0, 1.0 / span, float(y[0])]
    if name in ("Power", "Power (linear regression)"):
        ok = (x > 0) & (y > 0)
        if ok.sum() >= 2:
            b, loga = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
            return [float(np.exp(loga)), float(b)]
        return [1.0, 1.0]
    if name == "Log":
        return [(ymax - ymin) / max(np.log(max(xmax, 1e-9) / max(xmin, 1e-9)), 1e-9) if xmin > 0 else 1.0, 1.0]
    if name == "Log2":
        c = xmin - 0.1 * span
        b, a = np.polyfit(np.log(x - c), y, 1)
        return [float(a), float(b), float(c)]
    if name == "Gaussian":
        return [ymin, ymax, float(x[int(np.argmax(y))]), span / 4]
    if name == "Gaussian (no offset)":
        xpeak = float(x[int(np.argmax(y))])
        above = x[y >= ymax / 2]  # width from the full width at half maximum, when the data resolves it
        fwhm = float(above.max() - above.min()) if above.size >= 2 else 0.0
        return [ymax or 1.0, xpeak, (fwhm / 2.3548) if fwhm > 0 else span / 4]
    if name in ("Sigmoid (Boltzmann)", "Error Function"):
        mid = (ymin + ymax) / 2
        c = float(x[int(np.argmin(np.abs(y - mid)))])
        if name == "Error Function":
            rising = y[-1] >= y[0]
            return [mid, (ymax - ymin) / 2 * (1 if rising else -1), c, span / 5]
        return [float(y[0]), float(y[-1]), c, span / 10 if y[-1] >= y[0] else -span / 10]
    if name == "Rodbard":
        return [float(y[0]), 1.0, float(np.median(x)) or 1.0, float(y[-1])]
    if name == "Rational":
        b = (y[-1] - y[0]) / span
        return [float(y[0]), float(b), 0.0]
    if name == "Michaelis-Menten":
        half = ymax / 2
        return [ymax or 1.0, float(x[int(np.argmin(np.abs(y - half)))]) or 1.0]
    if name == "Gamma Variate":
        # ImageJ: a = first x, peak position gives c·d; start with c = 2 and d from the peak
        xpeak = float(x[int(np.argmax(y))])
        a = xmin - 0.01 * span
        d = max((xpeak - a) / 2.0, 1e-3)
        b = (ymax or 1.0) / max((xpeak - a) ** 2 * np.exp(-(xpeak - a) / d), 1e-9)
        return [a, float(b), 2.0, d]
    if name == "Chapman-Richards":
        return [ymax or 1.0, 3.0 / span, 1.5]
    if name in ("Rodbard (NIH Image)", "Inverse Rodbard"):
        a, d = xmin - 0.05 * span, xmax + 0.05 * span
        ratio = (x - a) / (d - x)
        ok = (ratio > 0) & (y > 0)
        if ok.sum() >= 2:
            slope, logc = np.polyfit(np.log(ratio[ok]), np.log(y[ok]), 1)
            if np.isfinite(slope) and slope != 0:
                return [a, float(1.0 / slope), float(np.exp(logc)), d]
        return [a, 1.0, float(np.median(y)) or 1.0, d]
    return [1.0] * FIT_FUNCTIONS[name][1]


def _custom_starts(fn: Callable[..., np.ndarray], n: int, x: np.ndarray, y: np.ndarray, keep: int = 6, samples: int = 240) -> list[list[float]]:
    """Starting vectors for a custom equation: all ones (ImageJ's default), a few typical shapes and random
    combinations of data-derived magnitudes (offsets, amplitudes, x positions, widths, rates), pre-screened by their
    sum of squared residuals so that only the ``keep`` most promising ones go to the optimiser."""
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    span = (xmax - xmin) or 1.0
    xpeak, xtrough = float(x[int(np.argmax(y))]), float(x[int(np.argmin(y))])
    slope = float((y[-1] - y[0]) / span)
    shapes = [
        [1.0] * 6,
        [ymax or 1.0, 1.0 / span, ymin, xpeak, span / 4, 1.0],   # amplitude, rate, offset, centre, width
        [ymin, ymax, xpeak, span / 4, 1.0, 1.0],                  # low, high, centre, width
        [float(y[0]), slope, 0.0, 1.0, 1.0, 1.0],                 # intercept, slope
        [ymax or 1.0, -1.0 / span, ymin, 1.0, 1.0, 1.0],          # decaying variant
        [ymin, ymax - ymin, span / 3, xpeak, 1.0, 1.0],           # offset, amplitude, length scale
    ]
    magnitudes = np.array([1.0, 0.0, -1.0, ymin, ymax, ymax - ymin, (ymax + ymin) / 2, float(y[0]), float(y[-1]), xmin, xmax, xpeak, xtrough,
                           (xmin + xmax) / 2, span, span / 4, span / 10, 1.0 / span, -1.0 / span, 3.0 / span, slope, 0.5, 2.0], dtype=np.float64)
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    rng = np.random.default_rng(0)
    candidates = [np.asarray(sh[:n], dtype=np.float64) for sh in shapes]
    candidates += [magnitudes[rng.integers(0, magnitudes.size, size=n)] for _ in range(samples)]
    scored: list[tuple[float, list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    for cand in candidates:
        key = tuple(float(v) for v in cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            with np.errstate(all="ignore"):
                pred = np.asarray(fn(x, *cand), dtype=np.float64)
                sse = float(np.sum((y - pred) ** 2))
        except Exception:
            continue
        scored.append((sse if np.isfinite(sse) else math.inf, list(key)))
    scored.sort(key=lambda t: t[0])
    starts = [v for _, v in scored[:keep]]
    ones = [1.0] * n
    if ones not in starts:
        starts.append(ones)
    return starts


def _perturbed(p0: list[float], restarts: int, seed: int = 0) -> list[list[float]]:
    """``restarts`` random variations of p0 (log-normal scale, occasional sign flip) for the multi-start search."""
    if restarts <= 0:
        return []
    rng = np.random.default_rng(seed)
    base = np.asarray(p0, dtype=np.float64)
    out = []
    for _ in range(int(restarts)):
        scale = np.exp(rng.normal(0.0, 0.7, size=base.size))
        flip = np.where(rng.random(base.size) < 0.15, -1.0, 1.0)
        v = np.where(base != 0, base * scale * flip, rng.normal(0.0, 1.0, size=base.size))
        out.append([float(a) for a in v])
    return out


def _least_squares(fn: Callable[..., np.ndarray], x: np.ndarray, y: np.ndarray, starts: list[list[float]]) -> tuple[list[float] | None, float]:
    """Levenberg–Marquardt from every start; returns the parameters with the smallest sum of squared residuals."""
    import warnings
    from scipy.optimize import OptimizeWarning, curve_fit
    best: list[float] | None = None
    best_sse = math.inf
    for p0 in starts:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)  # 'covariance could not be estimated' – only the parameters are used
                warnings.simplefilter("ignore", RuntimeWarning)
                params, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
                params = [float(p) for p in params]
                pred = np.asarray(fn(x, *params), dtype=np.float64)
            sse = float(np.sum((y - pred) ** 2))
        except Exception:
            continue
        if np.isfinite(sse) and sse < best_sse:
            best, best_sse = params, sse
    return best, best_sse


# ---------------------------------------------------------------------------------------- fitting
def fit(name: str, x: Any, y: Any, equation: str | None = None, initial_params: Any = None, restarts: int = 0) -> dict[str, Any]:
    """Fit `name` to the data. Returns {function, formula, params (list), names, r2, n, sse, rms, sd, r2_adj}.

    ``name="Custom"`` (or any name with ``equation`` given) fits the user-defined ``equation`` (see CustomEquation).
    ``initial_params`` (sequence or text) overrides the starting values; ``restarts`` adds that many random
    perturbations of the start to the multi-start search (the best sum of squared residuals wins).
    """
    name = canonical_name(name)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if isinstance(initial_params, str):
        initial_params = parse_numbers(initial_params) if initial_params.strip() else None
    p_init = [float(p) for p in initial_params] if initial_params is not None and len(initial_params) else None
    custom: CustomEquation | None = None
    if equation is not None and str(equation).strip() or name == CUSTOM:
        if equation is None or not str(equation).strip():
            raise ValueError("Custom fit: enter an equation such as y = a*exp(-b*x) + c")
        custom = compile_equation(str(equation))
        name, formula, n, fn = CUSTOM, custom.formula, custom.n, custom
        names = list(custom.params)
    elif name in FIT_FUNCTIONS:
        formula, n, fn = FIT_FUNCTIONS[name]
        names = list(PARAM_NAMES[:n])
    else:
        raise ValueError(f"Unknown fit function {name!r}. Choose one of: {', '.join(FIT_NAMES)} or Custom")
    if x.size < n:
        raise ValueError(f"{name} needs at least {n} data points (got {x.size})")
    if p_init is not None and len(p_init) != n:
        raise ValueError(f"{name} has {n} parameters ({', '.join(names)}) but {len(p_init)} initial values were given")
    if custom is not None:
        starts = [p_init] if p_init is not None else _custom_starts(fn, n, x, y)
        starts = starts + _perturbed(starts[0], restarts)
        params, _ = _least_squares(fn, x, y, starts)
        if params is None:
            params = list(starts[0])
    elif "Polynomial" in name or name == "Straight Line":
        coef = np.polyfit(x, y, n - 1)[::-1]  # ascending powers
        params = [float(c) for c in coef]
    elif name == "Exponential (linear regression)":
        pos = y > 0
        if pos.sum() < 2:
            raise ValueError("Exponential (linear regression) needs at least 2 points with y > 0")
        b, loga = np.polyfit(x[pos], np.log(y[pos]), 1)
        params = [float(np.exp(loga)), float(b)]
    elif name == "Power (linear regression)":
        pos = (x > 0) & (y > 0)
        if pos.sum() < 2:
            raise ValueError("Power (linear regression) needs at least 2 points with x > 0 and y > 0")
        b, loga = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
        params = [float(np.exp(loga)), float(b)]
    elif name == "Rodbard (NIH Image)":
        # ImageJ's RODBARD2: fit the Rodbard function with x and y swapped, report its a, b, c, d
        params = fit("Rodbard", y, x, initial_params=p_init, restarts=restarts)["params"]
    else:
        p0 = p_init if p_init is not None else _initial_guess(name, x, y)
        params, _ = _least_squares(fn, x, y, [p0] + _perturbed(p0, restarts))
        if params is None:
            params = [float(p) for p in p0]
        if name == "Log" and x.size and (np.all(x > 0) or np.all(x < 0)) and params[1] * float(x[0]) < 0:
            # _log evaluates ln(|b·x|) (keeps scipy's LM free of NaN), so the optimiser cannot tell the sign of b;
            # report the sign for which the displayed formula y = a·ln(b·x) is defined on the data (ImageJ's
            # CurveFitter never returns b·x < 0). Predictions and R² are unchanged because the model is even in b.
            params[1] = -params[1]
    with np.errstate(all="ignore"):
        pred = np.asarray(fn(x, *params), dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res < 1e-12 else 0.0)
    n_pts = int(x.size)
    rms = math.sqrt(ss_res / n_pts) if n_pts else float("nan")
    sd = math.sqrt(ss_res / (n_pts - n)) if n_pts > n else float("nan")  # ImageJ's "Standard deviation"
    r2_adj = 1.0 - (1.0 - r2) * (n_pts - 1) / (n_pts - n - 1) if n_pts > n + 1 else r2  # ImageJ's "Adjusted R^2"
    out = {"function": name, "formula": formula, "params": params, "names": names, "r2": float(r2), "n": n_pts,
           "sse": ss_res, "rms": float(rms), "sd": float(sd), "r2_adj": float(r2_adj)}
    if custom is not None:
        out["equation"] = custom.text
    return out


def evaluate(name: str, params: list[float], x: Any, equation: str | None = None) -> np.ndarray:
    """Evaluate a fitted function at x (``equation`` is required for name == "Custom")."""
    if name == "Uncalibrated OD":
        v = np.asarray(x, dtype=np.float64)
        return np.log10(255.0 / np.maximum(255.0 - v, 0.5))
    name = canonical_name(name)
    xs = np.asarray(x, dtype=np.float64)
    if name == CUSTOM or (equation is not None and str(equation).strip() and name not in FIT_FUNCTIONS):
        if equation is None or not str(equation).strip():
            raise ValueError("A custom fit needs its equation to be evaluated")
        return compile_equation(str(equation))(xs, *params)
    if name not in FIT_FUNCTIONS:
        raise ValueError(f"Unknown fit function {name!r}")
    fn = FIT_FUNCTIONS[name][2]
    with np.errstate(all="ignore"):
        out = np.asarray(fn(xs, *params), dtype=np.float64)
        if name == "Log2" and len(params) == 3:
            out = np.where(xs - params[2] > 0, out, np.nan)  # ln(x − c) is undefined left of c (ImageJ returns NaN there)
    return out


def residuals(name: str, params: list[float], x: Any, y: Any, equation: str | None = None) -> np.ndarray:
    """y − f(x) for the fitted function (the data of ImageJ's residuals plot)."""
    return np.asarray(y, dtype=np.float64) - evaluate(name, params, x, equation=equation)


def parse_numbers(text: str) -> np.ndarray:
    parts = [p for p in re.split(r"[\s,;]+", str(text).strip()) if p]
    return np.array([float(p) for p in parts], dtype=np.float64)
