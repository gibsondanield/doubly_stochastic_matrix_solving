# Wine cellar riddle: strategies and probability grids

Code for the blog post [Reverse-engineering the wine cellar](https://gibsondanield.github.io/blog/wine_cellar/),
a follow-up to Chris KW's [Sixteen bottles of wine riddle](https://chriskw.xyz/2025/08/11/Wine/).

There are 2<sup>n</sup> bottles, each with a distinct n-bit code, shuffled. One measurement reveals one bit of
one bottle. The goal is to identify every bottle in as few measurements as possible.

## Results (8 bottles)

| strategy | mean measurements |
|---|---|
| measure everything | 24.00 |
| constraint propagation | 19.55 |
| greedy, unweighted entropy score | 15.90 |
| **greedy, measure the bit closest to 50/50, exact probabilities** | **15.43** (200 cellars) |
| floor: log<sub>2</sub>(8!) | 15.3 |

## What's here

- `main.py`: the cellar, and a script that runs every strategy on shuffled cellars.
- `algorithms.py`: the strategies. `GreedyClosestToHalfExact` is the best one.
- `matrix_utilities.py`: the probability grid P[bottle, code]:
  - `exact_probability_matrix`: exact, as ratios of permanents (Ryser's formula). Practical up to 16 bottles.
  - `solve_probability_matrix`: the LP approach. It finds exactly which cells are forced, but its probabilities
    are only an approximation unless symmetry pins them down.
  - `brute_force_probabilities`: enumerates every arrangement; used to check the other two.
  - `permanent_upper_bound`: the Bregman–Minc bound.
- `inequality_solving.py`: LP helpers, plus the earlier symbolic (SymPy) solver.
- `figures/`: the script for the post's figure.

## Running

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python main.py 3 20          # 8 bottles, 20 shuffled cellars, all strategies (a few minutes)
uv run python main.py 3 200 --fast  # skip the slow greedies
uv run python -m unittest
```

Four tests are marked as expected failures. They compare the LP's probabilities with brute force, and
document that the LP is only exact when the constraints determine the answer.

## License

MIT
