"""State populations from many short trajectory segments: a small Markov state model.

Adaptive sampling restarts trajectory segments from chosen frames, for example from rarely visited
states. Counting the time in each state then gives wrong populations, because the starting frames
were chosen on purpose. A Markov state model does not need equilibrium starting points: it counts
the transitions between states at a lag time, estimates the transition probabilities, and takes
the populations from the stationary distribution of that model (Prinz et al., J. Chem. Phys. 2011,
134, 174105; Bowman et al., J. Chem. Phys. 2009, 131, 124101).

The states here are the cores of each quantity (chi1 g+/t/g-, or near/far). Frames between cores
keep the last core visited, as in core-set Markov models (Buchete and Hummer, J. Phys. Chem. B
2008, 112, 6057), which keeps the model Markovian at short lag times.

- Transition matrix: reversible maximum likelihood estimate (detailed balance), by the
  fixed-point iteration of Bowman et al. 2009 / Prinz et al. 2011.
- Only states connected in both directions count. A state entered but never left (or left but
  never entered) gets no population; the result says so.
- Error: bootstrap over segments (segments are the independent pieces), never below the
  error floor of uncertainty.floor_from_transitions.
"""

from __future__ import annotations

import numpy as np

from .uncertainty import floor_from_transitions


def core_fill(states: np.ndarray) -> np.ndarray:
    """Frames between cores (-1) keep the last core; frames before the first core stay -1."""
    s = np.asarray(states, dtype=int).copy()
    last = -1
    for i, x in enumerate(s):
        if x < 0:
            s[i] = last
        else:
            last = x
    return s


def count_matrix(segments: list[np.ndarray], n_states: int, lag: int = 1) -> np.ndarray:
    c = np.zeros((n_states, n_states))
    for seg in segments:
        s = core_fill(seg)
        a, b = s[:-lag], s[lag:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(c, (a[ok], b[ok]), 1.0)
    return c


def connected_states(c: np.ndarray) -> list[int]:
    """The strongly connected set of states (each reachable from each other) with the most counts."""
    best, best_counts = [], -1.0
    for start in range(len(c)):
        if c[start].sum() + c[:, start].sum() == 0:
            continue
        comp = sorted(_reach(c > 0, start) & _reach((c > 0).T, start))
        counts = c[np.ix_(comp, comp)].sum()
        if counts > best_counts:
            best, best_counts = comp, counts
    return best


def _reach(adj: np.ndarray, start: int) -> set[int]:
    seen, todo = {start}, [start]
    while todo:
        i = todo.pop()
        for j in np.nonzero(adj[i])[0]:
            if int(j) not in seen:
                seen.add(int(j))
                todo.append(int(j))
    return seen


def reversible_stationary(c: np.ndarray, iterations: int = 10000, tol: float = 1e-12) -> np.ndarray:
    """Stationary distribution of the reversible maximum likelihood transition matrix of counts c."""
    cs = c + c.T
    x = cs / cs.sum()
    ci = c.sum(axis=1)
    for _ in range(iterations):
        xi = x.sum(axis=1)
        denom = ci[:, None] / np.where(xi[:, None] > 0, xi[:, None], 1) + ci[None, :] / np.where(xi[None, :] > 0,
                                                                                               xi[None, :], 1)
        new = np.where(denom > 0, cs / np.where(denom > 0, denom, 1), 0.0)
        new /= new.sum()
        if np.abs(new - x).max() < tol:
            x = new
            break
        x = new
    pi = x.sum(axis=1)
    return pi / pi.sum()


def populations(segments: list[np.ndarray], names: list[str], lag: int = 1) -> tuple[dict, list[str]]:
    """Populations of each state from the Markov model, and the states left out (not connected)."""
    n = len(names)
    c = count_matrix(segments, n, lag)
    keep = connected_states(c)
    pops = {name: 0.0 for name in names}
    if not keep:
        return pops, list(names)
    pi = reversible_stationary(c[np.ix_(keep, keep)])
    for k, p in zip(keep, pi):
        pops[names[k]] = float(p)
    left_out = [names[k] for k in range(n) if k not in keep and (c[k].sum() + c[:, k].sum()) > 0]
    return pops, left_out


def populations_with_errors(segments: list[np.ndarray], names: list[str], lag: int = 1, n_boot: int = 200,
                            seed: int = 0) -> tuple[dict, dict, list[str]]:
    """Populations, bootstrap errors over segments (with the error floor), and the states left out."""
    from .diagnose import count_transitions

    pops, left_out = populations(segments, names, lag)
    rng = np.random.default_rng(seed)
    samples = []
    n = len(segments)
    for _ in range(n_boot if n > 1 else 0):
        pick = [segments[i] for i in rng.integers(0, n, n)]
        p, _ = populations(pick, names, lag)
        samples.append([p[k] for k in names])
    spread = np.std(np.array(samples), axis=0, ddof=1) if len(samples) > 1 else np.zeros(len(names))
    changes = sum(count_transitions(s) for s in segments)
    errs = {k: float(max(spread[i], floor_from_transitions(pops[k], changes, n))) for i, k in enumerate(names)}
    return pops, errs, left_out
