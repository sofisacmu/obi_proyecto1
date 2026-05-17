"""
Scenario generation utilities for the stochastic power network design project.

This module contains only reusable functions so it can be imported from other
notebooks, especially the Project 2 notebook for algorithms, SAA, VSS/EVPI,
and extended backtesting.

Typical use
-----------
from scenarios_utils import (
    load_edges_from_csv,
    load_nodes_from_csv,
    generate_scenarios_table,
    build_scenario_data,
)

edges = load_edges_from_csv("edges.csv")
nodes = load_nodes_from_csv("nodes.csv")
nodes = nodes[nodes["instance"] == 69].reset_index(drop=True)

dfD, dfU, dfc, dfp, dfG = generate_scenarios_table(
    nodes, edges, N=50, seed=1, K=None
)

omegas, p_omega, D, U_ew, c_ew, G_iw = build_scenario_data(
    dfD, dfU, dfc, dfG, dfp
)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def load_edges_from_csv(path_edges: str) -> pd.DataFrame:
    """
    Load the network arc data.

    Expected CSV columns:
    - node_id1: origin node
    - node_id2: destination node
    - b: base operational cost proxy
    - f_max: base line capacity

    Returns
    -------
    pd.DataFrame
        Columns include:
        - i: origin node
        - j: destination node
        - c_base: base operational cost
        - U_base: base arc capacity
        - e_id: unique arc identifier
    """
    edges = pd.read_csv(path_edges)

    required = {"node_id1", "node_id2", "b", "f_max"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing columns in edges file: {missing}")

    edges = edges.rename(
        columns={
            "node_id1": "i",
            "node_id2": "j",
            "b": "c_base",
            "f_max": "U_base",
        }
    )

    edges["e_id"] = np.arange(len(edges), dtype=int)

    # Enforce numerical types to avoid dtype issues later.
    edges["c_base"] = edges["c_base"].astype(float)
    edges["U_base"] = edges["U_base"].astype(float)

    return edges


def load_nodes_from_csv(path_nodes: str) -> pd.DataFrame:
    """
    Load the node data.

    Expected CSV columns:
    - node_id
    - d: base demand
    - p_min
    - p_max: maximum generation capacity
    - c_var
    - is_generator
    - energy_type
    - instance

    Returns
    -------
    pd.DataFrame
        Same information, with node_id renamed to node.
    """
    nodes = pd.read_csv(path_nodes)

    if "node_id" not in nodes.columns:
        raise ValueError("Missing column 'node_id' in nodes file.")

    nodes = nodes.rename(columns={"node_id": "node"})

    # Robust typing
    for col in ["d", "p_min", "p_max", "c_var"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce").fillna(0.0)

    if "is_generator" in nodes.columns:
        # Handles bools and strings such as "True"/"False".
        if nodes["is_generator"].dtype == object:
            nodes["is_generator"] = (
                nodes["is_generator"].astype(str).str.lower().isin(["true", "1", "yes"])
            )
        else:
            nodes["is_generator"] = nodes["is_generator"].astype(bool)

    return nodes


def sample_one_scenario_W(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    rng: np.random.Generator,
    p_crisis: float = 0.15,
    # Mediocristan
    dem_sigma: float = 0.10,
    cost_sigma: float = 0.05,
    p_outage: float = 0.15,
    cap_drop: float = 0.65,
    # Extremistan
    crisis_dem_mult_mu: float = 2.0,  # kept for compatibility; Pareto is used below
    crisis_dem_mult_sigma: float = 0.25,  # kept for compatibility
    crisis_cost_mult: float = 1.25,
    correlated_outage_nodes: Optional[Sequence[int]] = None,
    correlated_outage_prob: float = 0.65,
    pareto_alpha: float = 2.5,
    normal_gen_sigma: float = 0.05,
    crisis_gen_low: float = 0.30,
    crisis_gen_high: float = 1.00,
) -> Dict[str, object]:
    """
    Generate one scenario from the true uncertainty model W.

    A scenario contains:
    - D: demand by node
    - U: effective capacity by arc
    - c: operating cost by arc
    - G: available generation by node
    - crisis: binary indicator of the extreme regime

    Returns
    -------
    dict
        {
            "crisis": int,
            "D": DataFrame(node, D_w),
            "U": DataFrame(e_id, U_w),
            "c": DataFrame(e_id, c_w),
            "G": DataFrame(node, G_w),
        }
    """

    crisis = rng.random() < p_crisis

    # -------------------------
    # Demand D_iw
    # -------------------------
    D = nodes[["node"]].copy()
    D["D_base"] = nodes["d"].fillna(0.0).astype(float)

    if not crisis:
        demand_mult = rng.normal(1.0, dem_sigma, size=len(D))
        demand_mult = np.clip(demand_mult, 0.0, None)
    else:
        # Heavy-tailed shock: Pareto generates rare but large demand spikes.
        Y = rng.pareto(a=pareto_alpha, size=len(D))
        demand_mult = 1.0 + Y

    D["D_w"] = D["D_base"] * demand_mult
    D.loc[D["D_base"] == 0, "D_w"] = 0.0

    # -------------------------
    # Arc operating costs c_ew
    # -------------------------
    c = edges[["e_id", "c_base"]].copy()
    c["c_base"] = c["c_base"].astype(float)

    if not crisis:
        cost_mult = rng.normal(1.0, cost_sigma, size=len(c))
        cost_mult = np.clip(cost_mult, 0.1, None)
    else:
        cost_mult = np.ones(len(c)) * crisis_cost_mult

    c["c_w"] = c["c_base"] * cost_mult

    # -------------------------
    # Effective arc capacity U_ew
    # -------------------------
    U = edges[["e_id", "U_base", "i", "j"]].copy()
    U["U_base"] = U["U_base"].astype(float)

    outage = rng.random(len(U)) < p_outage
    U_mult = np.where(outage, cap_drop, 1.0)

    if crisis and correlated_outage_nodes is not None:
        if rng.random() < correlated_outage_prob:
            affected = U["i"].isin(correlated_outage_nodes) | U["j"].isin(
                correlated_outage_nodes
            )
            U_mult = np.where(affected, 0.0, U_mult)

    U["U_w"] = U["U_base"] * U_mult

    # -------------------------
    # Available generation G_iw
    # -------------------------
    G = nodes[["node"]].copy()
    G["G_base"] = nodes["p_max"].fillna(0.0).astype(float)

    if not crisis:
        gen_mult = rng.normal(1.0, normal_gen_sigma, size=len(G))
        gen_mult = np.clip(gen_mult, 0.5, 1.0)
    else:
        gen_mult = rng.uniform(crisis_gen_low, crisis_gen_high, size=len(G))

    G["G_w"] = G["G_base"] * gen_mult
    G.loc[G["G_base"] == 0, "G_w"] = 0.0

    return {
        "crisis": int(crisis),
        "D": D[["node", "D_w"]],
        "U": U[["e_id", "U_w"]],
        "c": c[["e_id", "c_w"]],
        "G": G[["node", "G_w"]],
    }


def generate_massive_W(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 123,
    correlated_outage_nodes: Optional[Sequence[int]] = None,
    **scenario_kwargs,
) -> List[Dict[str, object]]:
    """
    Generate a large Monte Carlo sample from W.

    Parameters
    ----------
    n_samples
        Number of scenarios to generate.
    seed
        Random seed for reproducibility.
    correlated_outage_nodes
        Nodes around which correlated outages may occur in crisis regimes.
    scenario_kwargs
        Extra parameters forwarded to sample_one_scenario_W.

    Returns
    -------
    list of dict
        Monte Carlo scenarios.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    for _ in range(n_samples):
        sc = sample_one_scenario_W(
            nodes,
            edges,
            rng,
            correlated_outage_nodes=correlated_outage_nodes,
            **scenario_kwargs,
        )
        scenarios.append(sc)

    return scenarios


def scenario_features(
    sc: Dict[str, object],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> np.ndarray:
    """
    Convert one scenario into a compact feature vector for clustering.

    Features:
    - total demand
    - average percentage of lost capacity
    - average operating cost
    - crisis indicator
    """
    Dtot = sc["D"]["D_w"].sum()

    U_base = edges.set_index("e_id")["U_base"].astype(float)
    U_eff = sc["U"].set_index("e_id")["U_w"].astype(float)

    cap_loss = 1.0 - (U_eff / U_base.replace(0, np.nan)).fillna(1.0)
    cap_loss_mean = cap_loss.mean()

    c_mean = sc["c"]["c_w"].mean()

    return np.array([Dtot, cap_loss_mean, c_mean, sc["crisis"]], dtype=float)


def reduce_to_K_scenarios(
    scenarios: List[Dict[str, object]],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    K: int = 20,
    seed: int = 123,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """
    Reduce a large scenario set to K representative scenarios using K-means.

    For each cluster, the representative scenario is the medoid, i.e.,
    the actual simulated scenario closest to the cluster centroid.

    Returns
    -------
    reps
        Representative scenarios.
    probs
        Probability of each representative scenario, equal to cluster size / total.
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if K > len(scenarios):
        raise ValueError("K cannot be larger than number of scenarios.")

    from sklearn.cluster import KMeans

    X = np.vstack([scenario_features(sc, nodes, edges) for sc in scenarios])

    km = KMeans(n_clusters=K, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    reps = []
    probs = []

    for k in range(K):
        idx = np.where(labels == k)[0]
        probs.append(len(idx) / len(scenarios))

        dists = np.linalg.norm(X[idx] - centers[k], axis=1)
        rep_idx = idx[np.argmin(dists)]
        reps.append(scenarios[rep_idx])

    return reps, np.array(probs, dtype=float)


def export_Wprime(
    reps: List[Dict[str, object]],
    probs: Sequence[float],
    out_prefix: Optional[str] = None,
    save_csv: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert representative scenarios into optimization tables.

    Tables:
    - dfD: omega, node, D_iw
    - dfU: omega, e_id, U_ew
    - dfc: omega, e_id, c_ew
    - dfp: omega, p_omega, crisis
    - dfG: omega, node, G_iw

    If save_csv=True, files are written using out_prefix.
    """
    rows_D, rows_U, rows_c, rows_p, rows_G = [], [], [], [], []

    for w, (sc, p) in enumerate(zip(reps, probs), start=1):
        rows_p.append({"omega": w, "p_omega": float(p), "crisis": bool(sc["crisis"])})

        for _, r in sc["D"].iterrows():
            rows_D.append({"omega": w, "node": int(r["node"]), "D_iw": float(r["D_w"])})

        for _, r in sc["U"].iterrows():
            rows_U.append({"omega": w, "e_id": int(r["e_id"]), "U_ew": float(r["U_w"])})

        for _, r in sc["c"].iterrows():
            rows_c.append({"omega": w, "e_id": int(r["e_id"]), "c_ew": float(r["c_w"])})

        for _, r in sc["G"].iterrows():
            rows_G.append({"omega": w, "node": int(r["node"]), "G_iw": float(r["G_w"])})

    dfD = pd.DataFrame(rows_D)
    dfU = pd.DataFrame(rows_U)
    dfc = pd.DataFrame(rows_c)
    dfp = pd.DataFrame(rows_p)
    dfG = pd.DataFrame(rows_G)

    # Explicit float typing avoids pandas LossySetitemError later.
    if not dfD.empty:
        dfD["D_iw"] = dfD["D_iw"].astype(float)
    if not dfU.empty:
        dfU["U_ew"] = dfU["U_ew"].astype(float)
    if not dfc.empty:
        dfc["c_ew"] = dfc["c_ew"].astype(float)
    if not dfG.empty:
        dfG["G_iw"] = dfG["G_iw"].astype(float)

    if save_csv:
        if out_prefix is None:
            out_prefix = "Wprime"
        dfD.to_csv(f"{out_prefix}_D.csv", index=False)
        dfU.to_csv(f"{out_prefix}_U.csv", index=False)
        dfc.to_csv(f"{out_prefix}_c.csv", index=False)
        dfp.to_csv(f"{out_prefix}_p.csv", index=False)
        dfG.to_csv(f"{out_prefix}_G.csv", index=False)

    return dfD, dfU, dfc, dfp, dfG


def generate_scenarios_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    N: int,
    seed: int = 123,
    K: Optional[int] = None,
    correlated_outage_nodes: Optional[Sequence[int]] = None,
    save_csv: bool = False,
    out_prefix: Optional[str] = None,
    **scenario_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate scenario tables for optimization.

    If K is None:
        returns N raw Monte Carlo scenarios with equal probabilities 1/N.

    If K is an integer:
        generates N raw scenarios and reduces them to K representative scenarios
        using K-means. The output probabilities are cluster frequencies.

    This is the main function to use from other notebooks.

    Examples
    --------
    # 50 raw scenarios
    dfD, dfU, dfc, dfp, dfG = generate_scenarios_table(nodes, edges, N=50, seed=1)

    # 1000 Monte Carlo scenarios reduced to 20 representatives
    dfD, dfU, dfc, dfp, dfG = generate_scenarios_table(
        nodes, edges, N=1000, K=20, seed=1
    )
    """
    if correlated_outage_nodes is None and "is_generator" in nodes.columns:
        correlated_outage_nodes = nodes.loc[nodes["is_generator"], "node"].tolist()

    raw = generate_massive_W(
        nodes,
        edges,
        n_samples=N,
        seed=seed,
        correlated_outage_nodes=correlated_outage_nodes,
        **scenario_kwargs,
    )

    if K is None:
        reps = raw
        probs = np.ones(N, dtype=float) / N
    else:
        reps, probs = reduce_to_K_scenarios(raw, nodes, edges, K=K, seed=seed)

    return export_Wprime(reps, probs, out_prefix=out_prefix, save_csv=save_csv)


def build_scenario_data(
    dfD: pd.DataFrame,
    dfU: pd.DataFrame,
    dfc: pd.DataFrame,
    dfG: pd.DataFrame,
    dfp: pd.DataFrame,
):
    """
    Convert scenario tables into dictionaries indexed by (omega, id).

    Returns
    -------
    omegas, p_omega, D, U_ew, c_ew, G_iw
    """
    omegas = dfp["omega"].tolist()
    p_omega = dict(zip(dfp["omega"], dfp["p_omega"]))

    D = {(int(r.omega), int(r.node)): float(r.D_iw) for _, r in dfD.iterrows()}
    U_ew = {(int(r.omega), int(r.e_id)): float(r.U_ew) for _, r in dfU.iterrows()}
    c_ew = {(int(r.omega), int(r.e_id)): float(r.c_ew) for _, r in dfc.iterrows()}
    G_iw = {(int(r.omega), int(r.node)): float(r.G_iw) for _, r in dfG.iterrows()}

    return omegas, p_omega, D, U_ew, c_ew, G_iw


def build_base_data(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    dfD: Optional[pd.DataFrame] = None,
    H_value: float = 50.0,
    budget: float = 2000.0,
) -> Dict[str, object]:
    """
    Build common network data used by optimization notebooks.

    Parameters
    ----------
    nodes, edges
        Network data.
    dfD
        Optional demand scenario table. If provided, nodes_list and demand_nodes
        are inferred from the scenario table. Otherwise, they are inferred from
        the node table.
    H_value
        Uniform reinforcement cost assigned to each arc.
    budget
        Investment budget.

    Returns
    -------
    dict
        base data with nodes_list, edges_list, demand_nodes, gen_nodes, topology,
        H_e, and budget.
    """
    edges_list = edges["e_id"].unique().tolist()

    if dfD is not None:
        nodes_list = dfD["node"].unique().tolist()
        demand_nodes = dfD.loc[dfD["D_iw"] > 0, "node"].unique().tolist()
    else:
        nodes_list = nodes["node"].unique().tolist()
        demand_nodes = nodes.loc[nodes["d"] > 0, "node"].unique().tolist()

    gen_nodes = nodes.loc[nodes["is_generator"], "node"].tolist()

    out_arcs = {i: edges.loc[edges["i"] == i, "e_id"].tolist() for i in nodes_list}
    in_arcs = {i: edges.loc[edges["j"] == i, "e_id"].tolist() for i in nodes_list}
    arc_i = dict(zip(edges["e_id"], edges["i"]))
    arc_j = dict(zip(edges["e_id"], edges["j"]))

    H_e = {e: float(H_value) for e in edges_list}

    return {
        "nodes_list": nodes_list,
        "edges_list": edges_list,
        "demand_nodes": demand_nodes,
        "gen_nodes": gen_nodes,
        "in_arcs": in_arcs,
        "out_arcs": out_arcs,
        "arc_i": arc_i,
        "arc_j": arc_j,
        "H_e": H_e,
        "budget": float(budget),
    }


def summarize_instance(nodes: pd.DataFrame, edges: pd.DataFrame, instance: Optional[int] = None):
    """
    Create small summary tables for reporting.
    """
    gens = nodes.loc[nodes["is_generator"], "node"].tolist()
    energy_summary = (
        nodes.loc[nodes["is_generator"]]
        .groupby("energy_type")["node"]
        .count()
        .reset_index()
        .rename(columns={"energy_type": "energy_type", "node": "count"})
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "instance",
                "total_nodes",
                "total_arcs",
                "total_generators",
                "generator_nodes",
            ],
            "value": [
                instance,
                len(nodes),
                len(edges),
                len(gens),
                str(gens),
            ],
        }
    )

    return summary, energy_summary


def summarize_scenarios(
    dfD: pd.DataFrame,
    dfU: pd.DataFrame,
    dfp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate scenario-level demand, capacity, crisis flag, and demand-capacity gap.
    """
    demanda_total = dfD.groupby("omega")["D_iw"].sum().reset_index()
    capacidad_total = dfU.groupby("omega")["U_ew"].sum().reset_index()
    crisis_df = dfp[["omega", "crisis"]].copy()

    scenario_table = (
        demanda_total.merge(crisis_df, on="omega").merge(capacidad_total, on="omega")
    )

    scenario_table["gap_demanda_capacidad"] = (
        scenario_table["D_iw"] - scenario_table["U_ew"]
    )

    return scenario_table.sort_values("D_iw", ascending=False).reset_index(drop=True)
