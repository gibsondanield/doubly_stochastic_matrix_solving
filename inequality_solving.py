"""
Vector inequality solver for one or more scalar symbols using SymPy.

Behavior:
- Builds the system 0 <= x[i] <= upper[i].
- If there is a single unique solution for all free symbols, return it and
  substitute into x to produce x_eval.
- If there are zero solutions, raise NoSolutionError.
- If there are multiple solutions, raise MultipleSolutionsError but also print
  insights about what is known.

Insights in multiple-solution cases:
- Variables uniquely determined by zero-width constraints.
- Ranges (bounds) of each remaining variable implied by inequalities.
- Detected dependent relationships among variables.

Scope:
- Exact uniqueness detection is fully supported for the univariate case.
- For multivariate inputs, uniqueness is inferred from zero-width constraints.
- Integer or rational coefficients are handled exactly.
"""
import sympy
import numpy as np
from scipy.optimize import linprog
import pandas as pd

def solve_lp(a,b, upper):
    # use both positive and negative to make equalities
    # A = pd.concat([a,-a])
    # B = pd.concat([pd.Series(b),-pd.Series(b)])
    #`A = np.vstack((a, -a))` and `B = np.hstack((b, -b))`.
    bounds = [[0,up] for up in upper]
    # Solve two LPs for each symbol to find min and max feasible values.
    results = {}
    for i, s in enumerate(a.columns):
        c = np.zeros(a.shape[1])
        c[i] = 1.0
        res_max = linprog(-c, A_eq=a, b_eq=b, bounds=bounds, method='highs')
        res_min = linprog(c, A_eq=a, b_eq=b, bounds=bounds, method='highs')
        if res_min.success and res_max.success:
            if abs(res_min.fun + res_max.fun) <.00001:
                results[s] = res_min.fun
                print('solved for',s, res_min.fun)
            else:
                print(s, 'interval' ,res_min.fun, -res_max.fun)
        else:
            print('failed to solve lp', s)
            # raise
            # results[s] = (None, None)
    print('ran lp', results)
    return results


def solve_vector_bounds_linprog(x, upper):
    """Alternative solver using scipy.optimize.linprog to find feasible bounds numerically."""
    syms = _sorted_free_symbols(sympy.Tuple(*x))
    if not syms:
        return np.array(x, dtype=float)

    # Convert symbolic expressions to linear form Ax <= b, -x <= 0.
    A = []
    b = []
    for xi, ui in zip(x, upper):
        expr = sympy.expand(xi)
        coeffs = [expr.coeff(s) for s in syms]
        const = expr - sum(c * s for c, s in zip(coeffs, syms))
        # xi >= 0 => -coeffs·syms <= const
        A.append([-float(c) for c in coeffs])
        b.append(float(const))
        # xi <= ui => coeffs·syms <= ui - const
        A.append([float(c) for c in coeffs])
        b.append(float(ui - const))

    # Initialize bounds (-inf, inf) for each variable.
    bounds = [(None, None)] * len(syms)

    # Solve two LPs for each symbol to find min and max feasible values.
    results = {}
    for i, s in enumerate(syms):
        c = np.zeros(len(syms))
        c[i] = 1.0
        res_max = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        res_min = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res_min.success and res_max.success:
            results[s] = (res_min.fun, -res_max.fun)
        else:
            results[s] = (None, None)
    return results

def solve_linear_system(A, b):
    # convert to sympy matrices
    A = sympy.Matrix(A)
    b = sympy.Matrix(b)
    sol = sympy.linsolve((A, b))
    print("Solution:", sol)
    if len(sol) == 0:
        raise NoSolutionError(f'not one solution {sol}')
    if len(sol) > 1:
        raise MultipleSolutionsError(f'multiple solutions {sol}')
    return sol.args[0]


class NoSolutionError(Exception):
    pass


class MultipleSolutionsError(Exception):
    pass


def _sorted_free_symbols(expr):
    return tuple(sorted(expr.free_symbols, key=lambda s: s.sort_key()))


def _solve_simple_or_and_ineq(ineq, syms):
    if ineq.free_symbols.isdisjoint(set(syms)):
        try:
            return sympy.S.Reals if bool(ineq) else sympy.S.EmptySet
        except TypeError:
            return sympy.S.EmptySet

    if isinstance(ineq, sympy.And):
        sol = sympy.S.Reals
        for arg in ineq.args:
            piece = sympy.S.Reals
            for s in syms:
                if s in arg.free_symbols:
                    piece = piece.intersect(
                        sympy.solve_univariate_inequality(arg, s, relational=False)
                    )
            sol = sol.intersect(piece)
        return sol

    if isinstance(ineq, sympy.core.relational.Relational):
        sol = sympy.S.Reals
        for s in syms:
            if s in ineq.free_symbols:
                sol = sol.intersect(
                    sympy.solve_univariate_inequality(ineq, s, relational=False)
                )
        return sol

    raise TypeError(f"Unsupported inequality type: {type(ineq)}")


def _implied_equalities_from_zero_width(x, upper):
    eqs = []
    for xi, ui in zip(x, upper):
        ui0 = sympy.simplify(ui)
        if ui0 == 0:
            eqs.append(sympy.Eq(sympy.simplify(xi), 0))
    return eqs


def _try_unique_solution_multivariate(x, upper):
    syms = _sorted_free_symbols(sympy.Tuple(*x))
    eqs = _implied_equalities_from_zero_width(x, upper)
    if not eqs:
        return None
    sol = sympy.solve(eqs, syms, dict=True)
    if isinstance(sol, list) and len(sol) == 1:
        assignment = sol[0]
        if all(s in assignment for s in syms):
            return assignment
    return None


def _analyze_partial_constraints(x, upper):
    syms = _sorted_free_symbols(sympy.Tuple(*x))
    eqs = _implied_equalities_from_zero_width(x, upper)
    known_vals = {}
    if eqs:
        known = sympy.solve(eqs, syms, dict=True)
        if known:
            known_vals = known[0]
    print("\n[Insight] Zero-width equalities:")
    for e in eqs:
        print("   ", e)
    print("[Insight] Known values:")
    if known_vals:
        for k, v in known_vals.items():
            print(f"   {k} = {v}")
    else:
        print("   None uniquely determined.")

    print("[Insight] Symbol ranges:")
    for s in syms:
        try:
            lo, hi = sympy.solve_univariate_inequality(0 <= s, s, relational=False), \
                     sympy.solve_univariate_inequality(s <= 1, s, relational=False)
            print(f"   {s}: possible range subset of intersection({lo}, {hi})")
        except Exception:
            print(f"   {s}: range undetermined")

    print("[Insight] Possible linear dependencies:")
    try:
        M, _ = sympy.linear_eq_to_matrix(_implied_equalities_from_zero_width(x, upper), syms)
        print("   Rank:", utils.rank(), "/", len(syms))
        if utils.rank() < len(syms):
            print("   Some variables remain dependent or free.")
    except Exception:
        print("   Could not compute dependencies.")


def solve_vector_bounds(x, upper):
    if len(x) != len(upper):
        raise ValueError("x and upper must have the same length")

    syms = _sorted_free_symbols(sympy.Tuple(*x))

    ineqs = [sympy.And(0 <= xi, xi <= ui) for xi, ui in zip(x, upper)]
    sol = sympy.S.Reals
    for ineq in ineqs:
        part = _solve_simple_or_and_ineq(ineq, syms)
        sol = sol.intersect(part)
        if sol == sympy.S.EmptySet or getattr(sol, 'is_empty', False):
            break

    if len(syms) == 1:
        s = syms[0]
        if sol == sympy.S.EmptySet or getattr(sol, 'is_empty', False):
            raise NoSolutionError("No solution for the given constraints")
        if isinstance(sol, sympy.FiniteSet):
            if len(sol) == 1:
                val = sympy.nsimplify(next(iter(sol)))
                subs_dict = {s: val}
                x_eval = sympy.Tuple(*[sympy.simplify(xi.subs(subs_dict)) for xi in x])
                return sol, syms, x_eval
            print("[Insight] Multiple discrete roots:", sol)
            raise MultipleSolutionsError("Multiple discrete solutions")
        print("[Insight] Continuum of solutions:", sol)
        raise MultipleSolutionsError("Continuum of solutions")

    if len(syms) > 1:
        assignment = _try_unique_solution_multivariate(x, upper)
        if assignment is not None:
            x_eval = sympy.Tuple(*[sympy.simplify(xi.subs(assignment)) for xi in x])
            return sympy.FiniteSet(sympy.Tuple(*[assignment[s] for s in syms])), syms, x_eval
        if sol == sympy.S.EmptySet or getattr(sol, 'is_empty', False):
            raise NoSolutionError("No solution for the given constraints")
        _analyze_partial_constraints(x, upper)
        raise MultipleSolutionsError("Multiple or continuum of solutions")

    if not syms:
        for xi, ui in zip(x, upper):
            if not (0 <= sympy.simplify(xi) <= sympy.simplify(ui)):
                raise NoSolutionError("Constant vector violates bounds")
        x_eval = sympy.Tuple(*[sympy.simplify(xi) for xi in x])
        return sympy.FiniteSet(), syms, x_eval

    raise MultipleSolutionsError("Unhandled case")



# if __name__ == "__main__":
    # tau0, tau1 = sympy.symbols('tau0 tau1', real=True)
    # x = sympy.Tuple(
    #     tau0 + tau1 - 1,
    #     -tau0 - tau1 + 2,
    #     2,
    #     tau1 + 1,
    #     -tau0 - tau1 + 3,
    #     tau0,
    #     1 - tau1,
    #     tau1,
    # )
    # upper = (2, 2, 5, 4, 3, 2, 2, 2)
    # try:
    #     solve_vector_bounds(x, upper)
    #     print("Test 1 — should have raised MultipleSolutionsError")
    # except MultipleSolutionsError:
    #     print("Test 1 — multiple solutions as expected")
    #
    # tau0 = sympy.symbols('tau0', real=True)
    # x = sympy.Tuple(2, tau0 + 1, -tau0, 4, 1 - tau0, tau0)
    # upper = (2, 1, 1, 5, 2, 2)
    # sol, syms, x_eval = solve_vector_bounds(x, upper)
    # expected = [2, 1, 0, 4, 1, 0]
    # np.testing.assert_array_almost_equal(np.array(x_eval, dtype=float), np.array(expected, dtype=float))
    # print("Test 2 — unique solution detected and substituted")
    #
    # taus = sympy.symbols('tau0:9', real=True)
    # tau0, tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8 = taus
    # x = sympy.Tuple(
    #     tau0 + tau1 + tau2 + tau4 + tau5 + tau7 + tau8 - 2,
    #     -tau0 - tau7 + 1,
    #     -tau1 - tau4 + 1,
    #     -tau2 - tau5 - tau8 + 1,
    #     1,
    #     tau6 + tau7 + tau8,
    #     tau3 + tau4 + tau5,
    #     -tau3 - tau6 + 1,
    #     -tau0 - tau1 - tau2 - tau4 - tau5 - tau7 - tau8 + 3,
    #     tau0,
    #     tau1,
    #     tau2,
    #     -tau3 - tau4 - tau5 + 1,
    #     tau3,
    #     tau4,
    #     tau5,
    #     -tau6 - tau7 - tau8 + 1,
    #     tau6,
    #     tau7,
    #     tau8,
    # )
    # upper = (1,) * len(x)
    # try:
    #     solve_vector_bounds(x, upper)
    #     print("Test 3 — should have raised MultipleSolutionsError")
    # except MultipleSolutionsError:
    #     print("Test 3 — multiple solutions as expected")

