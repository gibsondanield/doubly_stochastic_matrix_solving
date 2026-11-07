"""
Linear-programming helpers for the probability grid, plus the earlier symbolic solver.

LP (used by matrix_utilities.solve_probability_matrix):
- is_feasible: does a x = b, 0 <= x <= upper have any solution?
- solve_lp: per-variable min/max over that polytope, which variables are pinned down, and a
  feasible fill-in (the mean of the extreme points).

Symbolic (the first approach, kept for reference; see test_solve_inequalities1):
- solve_vector_bounds: for a vector x of SymPy expressions, solve 0 <= x[i] <= upper[i].
  Returns the unique solution if there is one, raises NoSolutionError if there are none, and
  MultipleSolutionsError (after logging what is known) if there are several. Exact uniqueness
  detection is supported for one free symbol; for several, uniqueness is inferred from
  zero-width constraints.
"""
import logging

import numpy as np
import sympy
from scipy.optimize import linprog

log = logging.getLogger(__name__)


class NoSolutionError(Exception):
    pass


class MultipleSolutionsError(Exception):
    pass


def is_feasible(a, b, upper):
    bounds = [(0, up) for up in upper]
    return linprog(np.zeros(a.shape[1]), A_eq=np.asarray(a, float), b_eq=np.asarray(b, float),
                   bounds=bounds, method='highs').success


def solve_lp(a, b, upper):
    '''For every variable, minimise and maximise it subject to a x = b, 0 <= x <= upper.

    Returns (fixed, x_avg):
      fixed  -- {column: value} for variables whose min == max (uniquely determined)
      x_avg  -- the mean of all the min/max solution vertices. Being a convex combination of
                feasible points it is itself feasible, so it is always a valid fill-in; for
                fixed variables it equals the determined value.
    '''
    A = np.asarray(a, float)
    B = np.asarray(b, float)
    bounds = [(0, up) for up in upper]
    fixed = {}
    vertices = []
    for i, s in enumerate(a.columns):
        c = np.zeros(A.shape[1])
        c[i] = 1.0
        res_max = linprog(-c, A_eq=A, b_eq=B, bounds=bounds, method='highs')
        res_min = linprog(c, A_eq=A, b_eq=B, bounds=bounds, method='highs')
        if not (res_min.success and res_max.success):
            raise NoSolutionError(f'LP infeasible while bounding {s}: {res_min.message}')
        vertices += [res_min.x, res_max.x]
        if abs(res_min.fun + res_max.fun) < 1e-5:
            fixed[s] = res_min.fun
            log.debug('solved for %s: %s', s, res_min.fun)
        else:
            log.debug('%s in interval [%s, %s]', s, res_min.fun, -res_max.fun)
    log.debug('LP fixed %d of %d variables', len(fixed), a.shape[1])
    return fixed, np.mean(vertices, axis=0)


# --- symbolic solver ---------------------------------------------------------------------------

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
    '''Logs what can be said when there is more than one solution.'''
    syms = _sorted_free_symbols(sympy.Tuple(*x))
    eqs = _implied_equalities_from_zero_width(x, upper)
    known_vals = {}
    if eqs:
        known = sympy.solve(eqs, syms, dict=True)
        if known:
            known_vals = known[0]
    log.info('zero-width equalities: %s', eqs or 'none')
    log.info('known values: %s', known_vals or 'none uniquely determined')
    if eqs:
        M, _ = sympy.linear_eq_to_matrix(eqs, syms)
        log.info('rank of implied equalities: %d / %d', M.rank(), len(syms))


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
            log.info('multiple discrete roots: %s', sol)
            raise MultipleSolutionsError("Multiple discrete solutions")
        log.info('continuum of solutions: %s', sol)
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
