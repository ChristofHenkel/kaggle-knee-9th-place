"""Small hyperparameter-search utilities used by the training runner.

The built-in search intentionally stays lightweight: it generates deterministic
quasi-random trials from a dictionary or uses an explicit list of trials. Larger
projects can still hand off orchestration to Optuna, Ray Tune, or MLflow jobs.
"""

from __future__ import annotations

import collections
import functools
import itertools
import math
from numpy import random
from typing import Any, Callable, Dict, List, Sequence, Text, Tuple, Union


SearchTrials = List[Dict[Text, Any]]
GeneratorFn = Callable[[float], Tuple[Text, Any]]


def generate_primes(n: int) -> List[int]:
    """Return odd prime numbers less than ``n`` using the Sieve of Sundaram."""
    half_m1 = int((n - 2) / 2)
    sieve = [0] * (half_m1 + 1)
    for outer in range(1, half_m1 + 1):
        inner = outer
        while outer + inner + 2 * outer * inner <= half_m1:
            sieve[outer + inner + (2 * outer * inner)] = 1
            inner += 1
    return [2 * i + 1 for i in range(1, half_m1 + 1) if sieve[i] == 0]


def _is_prime(n: int) -> bool:
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))


def _generate_dim(
    num_samples: int,
    base: int,
    per_dim_shift: bool,
    shuffled_seed_sequence: Sequence[int] | None,
) -> List[float]:
    """Generate one Van der Corput dimension in the requested prime base."""
    if base < 0 or not _is_prime(base):
        raise ValueError(f"Van der Corput base must be prime, got {base}.")

    rng = random.RandomState(base)
    if shuffled_seed_sequence is None:
        shuffled_seed_sequence = list(range(1, base))
        rng.shuffle(shuffled_seed_sequence)
        shuffled_seed_sequence = [0] + shuffled_seed_sequence

    dim_shift = rng.random_sample() if per_dim_shift else None
    dim_sequence = []
    for i in range(1, num_samples + 1):
        num = 0.0
        denominator = base
        while i:
            num += shuffled_seed_sequence[i % base] / denominator
            denominator *= base
            i //= base
        if per_dim_shift:
            num = math.fmod(num + dim_shift, 1.0)
        dim_sequence.append(num)
    return dim_sequence


def generate_sequence(
    num_samples: int,
    num_dims: int,
    skip: int = 100,
    per_dim_shift: bool = True,
    shuffle_sequence: bool = True,
    primes: Sequence[int] | None = None,
    shuffled_seed_sequence: Sequence[Sequence[int]] | None = None,
) -> List[Tuple[float, ...]]:
    """Generate ``num_samples`` points from a Halton sequence."""
    if skip < 0:
        raise ValueError(f"skip must be nonnegative, got {skip}.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if num_dims <= 0:
        raise ValueError(f"num_dims must be positive, got {num_dims}.")
    if primes is not None and len(primes) != num_dims:
        raise ValueError("primes must have one value per dimension.")
    if shuffled_seed_sequence is not None and len(shuffled_seed_sequence) != num_dims:
        raise ValueError("shuffled_seed_sequence must have one value per dimension.")

    if primes is None:
        primes = []
        attempts = 1
        while len(primes) < num_dims + 1:
            primes = generate_primes(1000 * attempts)
            attempts += 1
        primes = primes[-num_dims - 1 : -1]

    total_samples = num_samples + skip
    dimensions = []
    for dim in range(num_dims):
        seed_sequence = (
            None if shuffled_seed_sequence is None else shuffled_seed_sequence[dim]
        )
        if seed_sequence is not None and len(seed_sequence) != primes[dim]:
            raise ValueError(
                "Each shuffled seed sequence must match the corresponding prime base."
            )
        values = _generate_dim(
            total_samples,
            base=primes[dim],
            per_dim_shift=per_dim_shift,
            shuffled_seed_sequence=seed_sequence,
        )
        dimensions.append(values[skip:])

    sequence = list(zip(*dimensions))
    if shuffle_sequence:
        random.shuffle(sequence)
    return sequence


def _generate_double_point(
    name: Text, min_val: float, max_val: float, scaling: Text, halton_point: float
) -> Tuple[str, float]:
    if scaling not in {"linear", "log"}:
        raise ValueError(f"Unsupported floating-point scaling: {scaling}.")
    if scaling == "log":
        if min_val <= 0 or max_val <= 0:
            raise ValueError("Log-scaled search bounds must be positive.")
        value = min_val * math.exp(halton_point * math.log(max_val / min_val))
    else:
        value = halton_point * (max_val - min_val) + min_val
    return name, value


def _generate_discrete_point(
    name: str, feasible_points: Sequence[Any], halton_point: float
) -> Tuple[str, Any]:
    if len(feasible_points) == 0:
        raise ValueError(f"Discrete search space for {name} is empty.")
    index = min(int(math.floor(halton_point * len(feasible_points))), len(feasible_points) - 1)
    return name, feasible_points[index]


DiscretePoints = collections.namedtuple("DiscretePoints", "feasible_points")


def discrete(feasible_points: Sequence[Any]) -> DiscretePoints:
    """Declare a discrete set of feasible values for a sweep helper."""
    return DiscretePoints(feasible_points)


def interval(start: float, end: float) -> Tuple[float, float]:
    """Declare a numeric interval for a sweep helper."""
    return start, end


def loguniform(name: Text, range_endpoints: Tuple[float, float]) -> GeneratorFn:
    """Create a log-uniform generator for ``name``."""
    min_val, max_val = range_endpoints
    return functools.partial(_generate_double_point, name, min_val, max_val, "log")


def uniform(
    name: Text, search_points: Union[DiscretePoints, Tuple[float, float]]
) -> GeneratorFn:
    """Create a linear-uniform or discrete generator for ``name``."""
    if isinstance(search_points, DiscretePoints):
        return functools.partial(
            _generate_discrete_point, name, search_points.feasible_points
        )
    min_val, max_val = search_points
    return functools.partial(_generate_double_point, name, min_val, max_val, "linear")


def sweep(name: Text, feasible_points: DiscretePoints) -> SearchTrials:
    """Create explicit one-parameter trials for every feasible value."""
    return [{name: value} for value in feasible_points.feasible_points]


def product(sweeps: Sequence[SearchTrials]) -> SearchTrials:
    """Cartesian product of explicit sweep trial lists."""
    values_by_param = []
    for sweep_trials in sweeps:
        values_by_param.append([])
        for trial in sweep_trials:
            param_name, value = list(trial.items())[0]
            values_by_param[-1].append((param_name, value))
    return list(map(dict, itertools.product(*values_by_param)))


def zipit(
    generator_fns_or_sweeps: Sequence[Union[GeneratorFn, SearchTrials]],
    length: int,
) -> SearchTrials:
    """Zip quasi-random generators and explicit sweep lists into trial dicts."""
    sequence = generate_sequence(length, num_dims=len(generator_fns_or_sweeps))
    trials = []
    for trial_index in range(length):
        trial = {}
        for param_index, generator_or_sweep in enumerate(generator_fns_or_sweeps):
            if callable(generator_or_sweep):
                name, value = generator_or_sweep(sequence[trial_index][param_index])
            else:
                if trial_index >= len(generator_or_sweep):
                    return trials
                name, value = list(generator_or_sweep[trial_index].items())[0]
            trial[name] = value
        trials.append(trial)
    return trials


DictSearchSpace = Dict[str, Dict[str, Union[str, float, Sequence[Any]]]]
ListSearchSpace = List[Dict[str, Any]]


def generate_search(
    search_space: Union[DictSearchSpace, ListSearchSpace], num_trials: int
) -> List[collections.namedtuple]:
    """Generate namedtuple trials from a search-space specification.

    Dict spaces support continuous values with ``{"min", "max", "scaling"}``
    and discrete values with ``{"feasible_points": [...]}``. A list of dicts is
    treated as explicit trial definitions.
    """
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}.")
    if isinstance(search_space, dict):
        names = list(search_space.keys())
    elif isinstance(search_space, list):
        if not search_space:
            raise ValueError("Explicit search-space list must not be empty.")
        names = list(search_space[0].keys())
    else:
        raise TypeError("search_space must be a dict or a list of trial dicts.")

    trial_cls = collections.namedtuple("Hyperparameters", names)
    if isinstance(search_space, list):
        return [trial_cls(**trial) for trial in search_space[:num_trials]]

    generators = []
    for name, space in search_space.items():
        if "feasible_points" in space:
            generators.append(uniform(name, discrete(space["feasible_points"])))
        else:
            scaling = space.get("scaling", "linear")
            bounds = interval(float(space["min"]), float(space["max"]))
            generators.append(loguniform(name, bounds) if scaling == "log" else uniform(name, bounds))
    return [trial_cls(**trial) for trial in zipit(generators, num_trials)]
