from enum import Enum, auto
from dataclasses import dataclass
import hashlib
from copy import deepcopy
from itertools import product
from collections.abc import Iterator
from typing import TypeVar


import numpy as np
from numpy.random import Generator
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import linprog, OptimizeResult
from scipy.stats import dirichlet, expon, beta, gamma
from scipy.stats._multivariate import dirichlet_frozen
import ray


class QuotaType(Enum):
    """Enumeration for the different types of quotas"""

    NONE = auto()
    GTE20 = 0.2
    GTE30 = 0.3
    GTE40 = 0.4
    EQU50 = auto()


class GenderBias(Enum):
    """Enumeration for the different gender biases"""

    NONE = 0.0
    LOW = 0.1
    MEDIUM = 0.2
    HIGH = 0.3
    VERY_HIGH = 0.4
    EXTREME = 0.5


@dataclass
class Params:
    alpha_prefs: float
    alpha_caps: float
    n_roles: int
    n_persons: int
    total_cap: int
    rng: Generator


@dataclass
class Data:
    params: Params
    prefs: np.ndarray
    genders: np.ndarray
    caps: np.ndarray
    tvd: float

    def __hash__(self):
        prefs_hash = hashlib.sha256(self.prefs.tobytes()).hexdigest()
        genders_hash = hashlib.sha256(self.genders.tobytes()).hexdigest()
        caps_hash = hashlib.sha256(self.caps.tobytes()).hexdigest()
        return hash((prefs_hash, genders_hash, caps_hash, self.tvd))


@dataclass
class Experiment:
    data: Data
    quota: QuotaType
    gender_bias: GenderBias
    res: OptimizeResult

    @property
    def is_good(self) -> bool:
        return self.res.success

    @property
    def alloc(self) -> np.ndarray:
        return self.res.x.reshape(self.data.params.n_persons, self.data.params.n_roles)

    @property
    def n_alloc(self) -> int:
        return int(self.alloc.sum())

    @property
    def alloc_persons_perc(self) -> float:
        return self.n_alloc / self.data.params.n_persons

    @property
    def alloc_caps_perc(self) -> float:
        return self.n_alloc / self.data.params.total_cap

    @property
    def person_util(self) -> np.ndarray:
        return (self.data.prefs * self.alloc).sum(axis=1)

    @property
    def total_util(self) -> float:
        return self.person_util.sum()

    @property
    def total_util_biased(self) -> float:
        return -self.res.fun

    @property
    def g0_util(self) -> float:
        return (self.person_util * (1 - self.data.genders)).sum()

    @property
    def g1_util(self) -> float:
        return (self.person_util * self.data.genders).sum()

    @property
    def caps(self) -> np.ndarray:
        return self.data.caps

    @property
    def g1_role(self) -> np.ndarray:
        return (
            (self.alloc * self.data.genders.reshape(self.data.params.n_persons, 1))
            .sum(axis=0)
            .astype(int)
        )

    @property
    def g0_role(self) -> np.ndarray:
        return (
            (
                self.alloc
                * (1 - self.data.genders.reshape(self.data.params.n_persons, 1))
            )
            .sum(axis=0)
            .astype(int)
        )

    @property
    def g1_caps_perc(self) -> np.ndarray:
        return self.g1_role / self.data.caps

    @property
    def g0_caps_perc(self) -> np.ndarray:
        return self.g0_role / self.data.caps

    @property
    def g1_g0_perc(self) -> float:
        return self.g1_role / (self.g1_role + self.g0_role)


@dataclass
class Simulation:
    data: Data
    experiments: list[Experiment]

    @property
    def id(self) -> str:
        return str(abs(hash(self.data)))[:8]

    def __str__(self) -> str:
        p = self.data.params
        return f"Sim_{len(self.experiments)}_n_persons{p.n_persons}_n_roles{p.n_roles}_total_cap{p.total_cap}_a_prefs{p.alpha_prefs}_a_caps{p.alpha_caps}_id{self.id}"


def log_safe(x: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, epsilon))


def calc_tvd(p: np.ndarray, q: np.ndarray) -> float:
    "Total variation distance between two probability distributions"
    return 0.5 * np.sum(np.abs(p - q))


def sample_exp_dist(
    size: int, lambda_: float, rng: Generator | int | None = None
) -> np.ndarray:
    return expon.rvs(scale=1 / lambda_, size=size, random_state=rng)


def sample_gamma_dist(
    size: int, alpha: float, theta: float = 1.0, rng: Generator | int | None = None
) -> np.ndarray:
    return gamma.rvs(alpha, scale=theta, size=size, random_state=rng)


def gen_alphas(
    dim: int,
    alpha: float,
    tvd: float,
    n_candidates: int = 10_000,
    rng: Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha is the expectation value of the distribution as theta = 1"""
    assert 0 <= tvd < 1, "TVD target must be in the range [0, 1)."
    rng = np.random.default_rng(seed=rng)

    if tvd == 0:
        alpha = sample_gamma_dist(size=dim, alpha=alpha, rng=rng)
        return alpha, alpha

    def sample_alphas() -> np.ndarray:
        alpha_g1 = sample_gamma_dist(size=dim, alpha=alpha, rng=rng)
        alpha_g0 = sample_gamma_dist(size=dim, alpha=alpha, rng=rng)
        mean_g0 = alpha_g0 / np.sum(alpha_g0)
        mean_g1 = alpha_g1 / np.sum(alpha_g1)
        return calc_tvd(mean_g0, mean_g1), (alpha_g0, alpha_g1)

    candidates = [sample_alphas() for _ in range(n_candidates)]
    cand_tvds = np.array([cand[0] for cand in candidates])
    index = np.argmin(np.abs(cand_tvds - tvd))
    return candidates[index][1]


def gen_gender_prefs(
    num: int,
    dist_g0: dirichlet_frozen,
    dist_g1: dirichlet_frozen,
    rng: Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    half = num // 2
    rest = num - half
    preferences = np.vstack(
        (
            dist_g0.rvs(size=half, random_state=rng),
            dist_g1.rvs(size=rest, random_state=rng),
        )
    )
    genders = np.array([0] * half + [1] * rest, dtype=int)
    return preferences, genders


def stick_breaking(
    alpha: float, dim: int, rng: Generator | int | None = None
) -> np.ndarray:
    """
    Perform the stick-breaking process for a Dirichlet Process.

    alpha < 1: first component will be the largest with strong decay
    alpha = 1: beta is uniformly drawn from (0, 1)
    alpha > 1: components will be more evenly distributed

    Parameters:
        alpha (float): Concentration parameter of the Dirichlet Process.
        num_components (int): Number of components (or sticks) to generate.
        random_state (int, np.random.Generator, optional): Random state for reproducibility.

    Returns:
        np.ndarray: Weights generated by the stick-breaking process.
    """
    beta_samples = beta.rvs(1.0, alpha, size=dim, random_state=rng)
    remaining_stick = 1.0
    weights = []

    for sample in beta_samples:
        weight = remaining_stick * sample
        weights.append(weight)
        remaining_stick *= 1 - sample

    weights_arr = np.array(weights)
    return weights_arr / weights_arr.sum()  # normalize as we truncated the process


def break_by_weights(total: int, weights: np.ndarray) -> np.ndarray:
    """
    Break a total number of items into components according to the given weights.

    Parameters:
        total (int): Total number of items to break.
        weights (np.ndarray): Weights for each component.

    Returns:
        np.ndarray: Number of items in each component.
    """
    weights = weights / np.sum(weights)
    float_parts = weights * total
    int_parts = np.floor(float_parts).astype(int)
    remainder = total - int_parts.sum()
    residuals = float_parts - int_parts
    indices = np.argsort(-residuals)[:remainder]
    int_parts[indices] += 1
    assert int_parts.sum() == total
    return int_parts


def gen_capacities(
    n_cap: int, total_cap: int, alpha: float, rng: Generator | int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed=rng)
    while True:
        weights = stick_breaking(alpha, n_cap, rng)  # larger ones will be in front
        caps = break_by_weights(total_cap, weights)
        if np.all(caps > 0):
            break
    return caps


def allocate_roles(
    prefs: np.ndarray,
    genders: np.ndarray,
    caps: np.ndarray,
    quota: QuotaType,
    gender_bias: float,
) -> OptimizeResult:
    """Allocate the roles to the persons based on their preferences and the given quotas.

    Note that vector x is flattened, i.e. x = [x_11, x_12, ..., x_1n, x_21, ..., x_mn],
    where x_ij is the indicator variable for person i in role j.
    """
    n_persons, n_roles = prefs.shape
    assert n_roles == caps.size
    assert n_persons == genders.size

    # Define linear program
    ## objective function: minimize -preferences
    c = -(prefs + gender_bias * (1 - genders).reshape(-1, 1)).flatten()
    bounds = (0, 1)
    ## inequality constraints: sum of persons in roles <= capacities of roles
    A_ub_cap = np.zeros((n_roles, n_roles * n_persons))
    for i in range(n_roles):
        A_ub_cap[i, i::n_roles] = 1
    b_ub_cap = caps
    ## inequality constraints: each person gets at most one role
    A_ub_rol = np.zeros((n_persons, n_roles * n_persons))
    for i in range(n_persons):
        A_ub_rol[i, i * n_roles : (i + 1) * n_roles] = 1
    b_ub_rol = np.ones(n_persons)
    A_ub = np.vstack((A_ub_cap, A_ub_rol))
    b_ub = np.hstack((b_ub_cap, b_ub_rol))

    match quota:
        case QuotaType.NONE:
            pass
        case QuotaType.EQU50:
            # equality constraints: exactly 50% for each gender in each role
            A_ub_eq = np.zeros((2 * n_roles, n_roles * n_persons))
            for i in range(n_roles):
                A_ub_eq[2 * i, i::n_roles] = 0.5 - genders
                A_ub_eq[2 * i + 1, i::n_roles] = genders - 0.5
            b_ub_eq = np.zeros(2 * n_roles)
            b_ub_eq[::2] = 0.5
            b_ub_eq[1::2] = -0.5

            A_ub = np.vstack((A_ub, A_ub_eq))
            b_ub = np.hstack((b_ub, b_ub_eq))

        case QuotaType() as quota:
            # inequality constraints: at least x0% of gender 1 in each role
            A_ub_x0 = np.zeros((n_roles, n_roles * n_persons))
            for i in range(n_roles):
                A_ub_x0[i, i::n_roles] = quota.value - genders
            b_ub_x0 = np.zeros(n_roles)

            A_ub = np.vstack((A_ub, A_ub_x0))
            b_ub = np.hstack((b_ub, b_ub_x0))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=1)

    return res


def generate_data(
    alpha_prefs: float,
    alpha_caps: float,
    n_roles: int,
    n_persons: int,
    total_cap: int,
    tvd: float | None = None,
    rng: Generator | int | None = None,
) -> Data:
    params = Params(
        alpha_prefs=alpha_prefs,
        alpha_caps=alpha_caps,
        n_roles=n_roles,
        n_persons=n_persons,
        total_cap=total_cap,
        rng=deepcopy(rng),
    )
    rng = np.random.default_rng(seed=rng)
    tvd = tvd or rng.uniform(0, 1)
    assert 0 <= tvd <= 1

    alpha_g0, alpha_g1 = gen_alphas(dim=n_roles, alpha=alpha_prefs, tvd=tvd, rng=rng)
    prior_g0 = dirichlet(alpha_g0)
    prior_g1 = dirichlet(alpha_g1)
    prefs, genders = gen_gender_prefs(num=n_persons, dist_g0=prior_g0, dist_g1=prior_g1)
    tvd = calc_tvd(prior_g0.mean(), prior_g1.mean())
    caps = gen_capacities(n_cap=n_roles, total_cap=total_cap, alpha=alpha_caps, rng=rng)
    return Data(params=params, prefs=prefs, genders=genders, caps=caps, tvd=tvd)


def run_experiment(data: Data, quota: QuotaType, gender_bias: GenderBias) -> Experiment:
    res = allocate_roles(
        prefs=data.prefs,
        genders=data.genders,
        caps=data.caps,
        quota=quota,
        gender_bias=gender_bias.value,
    )
    return Experiment(data=data, quota=quota, gender_bias=gender_bias, res=res)


def run_simulation(
    alpha_prefs: float,
    alpha_caps: float,
    n_roles: int,
    n_persons: int,
    total_cap: int,
    n_sims: int = 1,
    rng: Generator | None = None,
) -> list[Simulation]:
    print("Running simulation with params: %s", locals())
    simulations = []
    with tqdm(
        total=n_sims * len(QuotaType) * len(GenderBias), desc="Simulations", leave=False
    ) as progress:
        for _ in range(n_sims):
            data = generate_data(
                alpha_prefs=alpha_prefs,
                alpha_caps=alpha_caps,
                n_roles=n_roles,
                n_persons=n_persons,
                total_cap=total_cap,
                rng=rng,
            )
            experiments = []
            for quota, gender_bias in product(QuotaType, GenderBias):
                experiments.append(
                    run_experiment(data=data, quota=quota, gender_bias=gender_bias)
                )
                progress.update(1)
            simulations.append(Simulation(data=data, experiments=experiments))
    return simulations


T = TypeVar("T")


def ray_pipeline(jobs: Iterator[T], n_jobs: int, n_parallel: int) -> Iterator[T]:
    """Run a pipeline of jobs with a given number of parallel tasks.

    Parameters:
        jobs (Iterator): Iterator of jobs to run. This triggers the computation.
        n_jobs (int): Total number of jobs to run.
        n_parallel (int): Number of parallel tasks to run.
    """
    rem_jobs = []
    with tqdm(total=n_jobs, desc="Ray Pipeline") as pbar:
        for job in jobs:
            rem_jobs.append(job)
            if len(rem_jobs) >= n_parallel:
                done_jobs, rem_jobs = ray.wait(rem_jobs, num_returns=1)
                pbar.update(1)
                yield ray.get(done_jobs[0])

        while len(rem_jobs) > 0:
            done_jobs, rem_jobs = ray.wait(rem_jobs, num_returns=1)
            pbar.update(1)
            yield ray.get(done_jobs[0])


def make_df(simulations: list[Simulation]) -> pd.DataFrame:
    rows = []
    for sim in simulations:
        for exp in sim.experiments:
            row = {
                "id": sim.id,
                "lambda": sim.data.params.alpha_prefs,
                "n_roles": sim.data.params.n_roles,
                "n_persons": sim.data.params.n_persons,
                "gender_bias": exp.gender_bias.name,
                "alpha_prefs": sim.data.params.alpha_prefs,
                "alpha_caps": sim.data.params.alpha_caps,
                "total_g0": sim.data.genders.size - sim.data.genders.sum(),
                "total_g1": sim.data.genders.sum(),
                "total_cap": sim.data.params.total_cap,
                "caps": sim.data.caps,
                "tvd": sim.data.tvd,
                "quota": exp.quota.name,
                "success": exp.is_good,
            }
            if exp.is_good:
                row |= {
                    "fun": exp.res.fun,
                    "total_util": exp.total_util,
                    "total_util_biased": exp.total_util_biased,
                    "g0_util": exp.g0_util,
                    "g1_util": exp.g1_util,
                    "g0_role": exp.g0_role,
                    "g0_caps_perc": exp.g0_caps_perc,
                    "g1_role": exp.g1_role,
                    "g1_caps_perc": exp.g1_caps_perc,
                    "g1_g0_perc": exp.g1_g0_perc,
                    "n_alloc": exp.n_alloc,
                    "alloc_persons_perc": exp.alloc_persons_perc,
                    "alloc_caps_perc": exp.alloc_caps_perc,
                }
            rows.append(row)
    df = pd.DataFrame(rows)

    def add_util_perc(df: pd.DataFrame, col: str, group_cols: list) -> pd.Series:
        none_util = (
            df[df["quota"] == QuotaType.NONE.name]
            .groupby(group_cols)[col]
            .first()  # Take the first value for each group
        )

        def compute_row(row):
            key = tuple(row[group_cols])
            return row[col] / none_util.get(key, float("nan"))

        return df.apply(compute_row, axis=1)

    if exp.is_good:
        group_cols = ["id", "gender_bias"]  # Define the columns to group by
        df["total_util_perc"] = add_util_perc(df, "total_util", group_cols)
        df["g0_util_perc"] = add_util_perc(df, "g0_util", group_cols)
        df["g1_util_perc"] = add_util_perc(df, "g1_util", group_cols)
    return df


def all_simulations(
    n_parallel: int, rng: Generator | int | None = None
) -> Iterator[Simulation]:
    alpha_prefs_set = [1, 5]
    alpha_caps_set = [5, 10]
    n_roles_set = [5, 10, 20]
    n_persons_set = [500, 1000, 2_000]
    total_cap_set = [500, 1_000, 2_000]
    n_simulations = 100

    rng = np.random.default_rng(seed=rng)
    all_combinations = list(
        product(
            alpha_prefs_set,
            alpha_caps_set,
            n_roles_set,
            n_persons_set,
            total_cap_set,
        )
    )
    jobs = (
        ray.remote(run_simulation).remote(
            alpha_prefs=alpha_prefs,
            alpha_caps=alpha_caps,
            n_roles=n_roles,
            n_persons=n_person,
            total_cap=total_cap,
            n_sims=n_simulations,
            rng=deepcopy(rng),
        )
        for (
            alpha_prefs,
            alpha_caps,
            n_roles,
            n_person,
            total_cap,
        ) in all_combinations
    )

    return ray_pipeline(jobs, n_jobs=len(all_combinations), n_parallel=n_parallel)


if __name__ == "__main__":
    import pickle

    ray.init(log_to_driver=False, _system_config={"local_fs_capacity_threshold": 0.99})
    all_dfs = []
    for simulations in all_simulations(n_parallel=32, rng=42):
        file_name = str(simulations[0])
        with open(f"simulations/{file_name}.pkl", "wb") as fh:
            pickle.dump(simulations, fh)
        all_dfs.append(make_df(simulations))

    df = pd.concat(all_dfs)
    df.to_csv("simulations.csv", index=False)
    ray.shutdown()
