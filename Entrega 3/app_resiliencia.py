# app_resiliencia.py
# Dashboard Streamlit para el MDP de resiliencia eléctrica del Proyecto 3.
# Ejecutar en la misma carpeta donde estén: scenarios_utils.py, edges.csv y nodes.csv
# Comando: streamlit run app_resiliencia.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

try:
    from scenarios_utils import load_edges_from_csv, load_nodes_from_csv
except Exception:
    load_edges_from_csv = None
    load_nodes_from_csv = None

# ============================================================
# Configuración general
# ============================================================
st.set_page_config(
    page_title="MDP Resiliencia Eléctrica",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    .metric-card {background:#f8fafc; padding:1rem; border-radius:0.8rem; border:1px solid #e5e7eb;}
    .small-note {font-size:0.9rem; color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Funciones base del modelo
# ============================================================

def generate_dynamic_scenarios(
    nodes,
    edges,
    T=12,
    n_paths=100,
    seed=123,
    p_crisis=0.10,
    dem_sigma=0.10,
    cost_sigma=0.05,
    crisis_dem_mult=2.0,
    crisis_cost_mult=2.0,
    p_outage=0.02,
    crisis_p_outage=0.20,
    cap_drop=0.70,
):
    rng = np.random.default_rng(seed)
    rows_D, rows_U, rows_c, rows_G, rows_info = [], [], [], [], []

    for path in range(n_paths):
        for t in range(1, T + 1):
            crisis = rng.random() < p_crisis

            if crisis:
                dem_mult = rng.lognormal(mean=np.log(crisis_dem_mult), sigma=0.25)
                cost_mult = crisis_cost_mult
                outage_prob = crisis_p_outage
            else:
                dem_mult = max(rng.normal(1.0, dem_sigma), 0.0)
                cost_mult = max(rng.normal(1.0, cost_sigma), 0.1)
                outage_prob = p_outage

            for _, r in nodes.iterrows():
                node = int(r["node"])
                d_base = float(r.get("d", 0.0))
                g_base = float(r.get("p_max", 0.0)) if bool(r.get("is_generator", False)) else 0.0

                rows_D.append({"path": path, "t": t, "node": node, "D_it": d_base * dem_mult})
                rows_G.append({"path": path, "t": t, "node": node, "G_it": g_base * max(rng.normal(1.0, 0.05), 0.0)})

            for _, r in edges.iterrows():
                e = int(r["e_id"])
                U_base = float(r.get("U_base", r.get("f_max", 100.0)))
                c_base = float(r.get("c_base", 1.0))

                failed = rng.random() < outage_prob
                U_mult = cap_drop if failed else 1.0

                rows_U.append({"path": path, "t": t, "e_id": e, "U_et": U_base * U_mult, "failed": failed})
                rows_c.append({"path": path, "t": t, "e_id": e, "c_et": c_base * cost_mult})

            rows_info.append({"path": path, "t": t, "crisis": crisis, "dem_mult": dem_mult, "cost_mult": cost_mult})

    return (
        pd.DataFrame(rows_D),
        pd.DataFrame(rows_U),
        pd.DataFrame(rows_c),
        pd.DataFrame(rows_G),
        pd.DataFrame(rows_info),
    )


def get_exogenous_info(path, t, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn):
    D_t = dfD_dyn[(dfD_dyn["path"] == path) & (dfD_dyn["t"] == t)].set_index("node")["D_it"].to_dict()
    U_t = dfU_dyn[(dfU_dyn["path"] == path) & (dfU_dyn["t"] == t)].set_index("e_id")["U_et"].to_dict()
    failed_t = dfU_dyn[(dfU_dyn["path"] == path) & (dfU_dyn["t"] == t)].set_index("e_id")["failed"].astype(int).to_dict()
    c_t = dfc_dyn[(dfc_dyn["path"] == path) & (dfc_dyn["t"] == t)].set_index("e_id")["c_et"].to_dict()
    G_t = dfG_dyn[(dfG_dyn["path"] == path) & (dfG_dyn["t"] == t)].set_index("node")["G_it"].to_dict()
    info_row = df_info_dyn[(df_info_dyn["path"] == path) & (df_info_dyn["t"] == t)].iloc[0]

    return {
        "D": D_t,
        "U": U_t,
        "failed": failed_t,
        "c": c_t,
        "G": G_t,
        "crisis": bool(info_row["crisis"]),
        "dem_mult": float(info_row["dem_mult"]),
        "cost_mult": float(info_row["cost_mult"]),
    }


def initialize_state(nodes, edges, budget=2000.0):
    return {
        "t": 0,
        "budget_remaining": float(budget),
        "reinforced": {int(e): 0 for e in edges["e_id"]},
        "capacity": {int(r["e_id"]): float(r.get("U_base", r.get("f_max", 100.0))) for _, r in edges.iterrows()},
        "failed": {int(e): 0 for e in edges["e_id"]},
        "flow": {int(e): 0.0 for e in edges["e_id"]},
        "generation_dispatch": {int(r["node"]): 0.0 for _, r in nodes.iterrows()},
        "cumulative_deficit": 0.0,
        "cumulative_cost": 0.0,
        "demand": {int(r["node"]): float(r.get("d", 0.0)) for _, r in nodes.iterrows()},
        "generation_available": {
            int(r["node"]): float(r.get("p_max", 0.0)) if bool(r.get("is_generator", False)) else 0.0
            for _, r in nodes.iterrows()
        },
        "crisis": False,
    }


def validate_operational_decisions(state, action, W, nodes, edges):
    edges_list = edges["e_id"].astype(int).tolist()
    nodes_list = nodes["node"].astype(int).tolist()
    flow, generation_dispatch = {}, {}

    for e in edges_list:
        f_e = float(action.get("flow", {}).get(e, 0.0))
        cap_e = max(float(state["capacity"].get(e, 0.0)), 0.0)
        if state["failed"].get(e, 0) == 1:
            cap_e = 0.0
        flow[e] = min(max(f_e, 0.0), cap_e)

    for i in nodes_list:
        g_i = float(action.get("generation_dispatch", {}).get(i, 0.0))
        G_i = max(float(W["G"].get(i, 0.0)), 0.0)
        generation_dispatch[i] = min(max(g_i, 0.0), G_i)

    return flow, generation_dispatch


def operating_model_from_action(state, action, W, nodes, edges):
    nodes_list = nodes["node"].astype(int).tolist()
    edges_list = edges["e_id"].astype(int).tolist()

    in_arcs = {int(i): edges.loc[edges["j"] == i, "e_id"].astype(int).tolist() for i in nodes_list}
    out_arcs = {int(i): edges.loc[edges["i"] == i, "e_id"].astype(int).tolist() for i in nodes_list}

    flow, generation_dispatch = validate_operational_decisions(state, action, W, nodes, edges)
    delta, served_by_node = {}, {}

    for i in nodes_list:
        demand_i = max(float(W["D"].get(i, 0.0)), 0.0)
        inflow_i = sum(flow[e] for e in in_arcs.get(i, []))
        outflow_i = sum(flow[e] for e in out_arcs.get(i, []))
        gen_i = generation_dispatch.get(i, 0.0)
        available_i = gen_i + inflow_i - outflow_i
        served_i = min(demand_i, max(available_i, 0.0))
        deficit_i = max(demand_i - served_i, 0.0)
        served_by_node[i] = served_i
        delta[i] = deficit_i

    operating_cost = sum(W["c"].get(e, 0.0) * flow[e] for e in edges_list)
    total_demand = sum(max(float(W["D"].get(i, 0.0)), 0.0) for i in nodes_list)
    total_deficit = sum(delta.values())
    served_demand = total_demand - total_deficit

    return {
        "served_demand": served_demand,
        "total_deficit": total_deficit,
        "operating_cost": operating_cost,
        "flow": flow,
        "generation_dispatch": generation_dispatch,
        "delta": delta,
        "served_by_node": served_by_node,
    }


def compute_period_cost(state, action, W, operating_result, H_e, repair_cost=25.0, pi=190.0):
    reinforcement_cost = sum(H_e[e] * action["reinforce"].get(e, 0) for e in H_e if state["reinforced"].get(e, 0) == 0)
    repair_cost_total = repair_cost * sum(action["repair"].get(e, 0) for e in H_e)
    operating_cost = operating_result.get("operating_cost", 0.0)
    deficit_cost = pi * operating_result.get("total_deficit", 0.0)
    total_cost = reinforcement_cost + repair_cost_total + operating_cost + deficit_cost
    return {
        "total_cost": total_cost,
        "reinforcement_cost": reinforcement_cost,
        "repair_cost": repair_cost_total,
        "operating_cost": operating_cost,
        "deficit_cost": deficit_cost,
    }


def transition(state, action, W, nodes, edges, H_e, delta_u=0.8, budget_floor=0.0, repair_cost=25.0, pi=190.0):
    next_state = state.copy()
    next_state["reinforced"] = state["reinforced"].copy()
    next_state["capacity"] = state["capacity"].copy()
    next_state["failed"] = state["failed"].copy()
    next_state["demand"] = W["D"].copy()
    next_state["generation_available"] = W["G"].copy()

    for e, reinforce_decision in action["reinforce"].items():
        if reinforce_decision == 1 and next_state["reinforced"].get(e, 0) == 0:
            if next_state["budget_remaining"] - H_e[e] >= budget_floor:
                next_state["reinforced"][e] = 1

    for e, repair_decision in action["repair"].items():
        if repair_decision == 1 and next_state["budget_remaining"] - repair_cost >= budget_floor:
            next_state["failed"][e] = 0

    for e in next_state["capacity"].keys():
        failed_e = W["failed"].get(e, 0)
        if failed_e == 1:
            next_state["failed"][e] = 1
        elif next_state["failed"].get(e, 0) != 1:
            next_state["failed"][e] = 0

        cap_exog = max(float(W["U"].get(e, 0.0)), 0.0)
        if next_state["failed"].get(e, 0) == 1:
            next_state["capacity"][e] = 0.0
        elif next_state["reinforced"].get(e, 0) == 1:
            next_state["capacity"][e] = cap_exog * (1.0 + delta_u)
        else:
            next_state["capacity"][e] = cap_exog

    operating_result = operating_model_from_action(next_state, action, W, nodes, edges)
    next_state["flow"] = operating_result["flow"]
    next_state["generation_dispatch"] = operating_result["generation_dispatch"]

    cost_breakdown = compute_period_cost(state, action, W, operating_result, H_e, repair_cost, pi)
    next_state["budget_remaining"] = max(
        next_state["budget_remaining"] - cost_breakdown["reinforcement_cost"] - cost_breakdown["repair_cost"],
        budget_floor,
    )
    next_state["cumulative_deficit"] = state["cumulative_deficit"] + operating_result["total_deficit"]
    next_state["cumulative_cost"] = state["cumulative_cost"] + cost_breakdown["total_cost"]
    next_state["crisis"] = W["crisis"]
    next_state["t"] = state["t"] + 1

    info = {
        "operating_result": operating_result,
        "cost_breakdown": cost_breakdown,
        "num_failed_edges": sum(next_state["failed"].values()),
        "num_reinforced_edges": sum(next_state["reinforced"].values()),
        "total_flow": sum(operating_result["flow"].values()),
        "total_generation_dispatch": sum(operating_result["generation_dispatch"].values()),
    }
    return next_state, info

# ============================================================
# Políticas y Q-learning
# ============================================================
ACTION_NAMES = ["do_nothing", "repair_failed", "reinforce_low_capacity", "repair_and_reinforce"]


def discretize_state(state):
    failed = sum(state["failed"].values())
    reinforced = sum(state["reinforced"].values())
    budget = state["budget_remaining"]
    deficit = state["cumulative_deficit"]
    crisis = int(state["crisis"])
    failed_bin = min(failed // 10, 5)
    reinforced_bin = min(reinforced // 5, 5)
    budget_bin = 2 if budget > 1500 else (1 if budget > 500 else 0)
    deficit_bin = 0 if deficit == 0 else (1 if deficit < 500 else 2)
    return (crisis, failed_bin, reinforced_bin, budget_bin, deficit_bin)


def build_action(action_name, state, W, nodes, edges, H_e, max_edges=3):
    edges_list = edges["e_id"].astype(int).tolist()
    nodes_list = nodes["node"].astype(int).tolist()
    action = {
        "reinforce": {e: 0 for e in edges_list},
        "repair": {e: 0 for e in edges_list},
        "flow": {e: 0.0 for e in edges_list},
        "generation_dispatch": {i: 0.0 for i in nodes_list},
    }

    for i in nodes_list:
        action["generation_dispatch"][i] = W["G"].get(i, 0.0)
    for e in edges_list:
        if state["failed"].get(e, 0) == 0:
            action["flow"][e] = 0.25 * state["capacity"].get(e, 0.0)

    if action_name in ["repair_failed", "repair_and_reinforce"]:
        failed_edges = [e for e in edges_list if state["failed"].get(e, 0) == 1]
        for e in failed_edges[:max_edges]:
            action["repair"][e] = 1

    if action_name in ["reinforce_low_capacity", "repair_and_reinforce"]:
        candidate_edges = [e for e in edges_list if state["reinforced"].get(e, 0) == 0]
        candidate_edges = sorted(candidate_edges, key=lambda e: state["capacity"].get(e, 0.0))
        for e in candidate_edges[:max_edges]:
            action["reinforce"][e] = 1

    return action


def no_action_policy(state, W, nodes, edges, H_e, params=None):
    return build_action("do_nothing", state, W, nodes, edges, H_e)


def q_learning_policy_fn(state, W, nodes, edges, H_e, params=None):
    Q = params.get("Q", {})
    s = discretize_state(state)
    values = [Q.get((s, a), 0.0) for a in ACTION_NAMES]
    best_action_name = ACTION_NAMES[int(np.argmin(values))]
    return build_action(best_action_name, state, W, nodes, edges, H_e)


def run_episode(policy_fn, path, T, nodes, edges, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn, H_e,
                budget=2000.0, delta_u=0.8, repair_cost=25.0, pi=190.0, policy_params=None):
    if policy_params is None:
        policy_params = {}
    state = initialize_state(nodes, edges, budget)
    history = []

    for t in range(1, T + 1):
        W = get_exogenous_info(path, t, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn)
        action = policy_fn(state, W, nodes, edges, H_e, policy_params)
        next_state, info = transition(state, action, W, nodes, edges, H_e, delta_u, 0.0, repair_cost, pi)
        history.append({
            "path": path,
            "t": t,
            "crisis": W["crisis"],
            "budget_remaining": next_state["budget_remaining"],
            "period_cost": info["cost_breakdown"]["total_cost"],
            "operating_cost": info["cost_breakdown"]["operating_cost"],
            "reinforcement_cost": info["cost_breakdown"]["reinforcement_cost"],
            "repair_cost": info["cost_breakdown"]["repair_cost"],
            "deficit_cost": info["cost_breakdown"]["deficit_cost"],
            "total_deficit": info["operating_result"]["total_deficit"],
            "served_demand": info["operating_result"]["served_demand"],
            "total_flow": info["total_flow"],
            "total_generation_dispatch": info["total_generation_dispatch"],
            "num_failed_edges": info["num_failed_edges"],
            "num_reinforced_edges": info["num_reinforced_edges"],
            "action_reinforce": [e for e, v in action["reinforce"].items() if v == 1],
            "action_repair": [e for e, v in action["repair"].items() if v == 1],
            "cumulative_cost": next_state["cumulative_cost"],
            "cumulative_deficit": next_state["cumulative_deficit"],
        })
        state = next_state
    return pd.DataFrame(history)


def q_learning_train(nodes, edges, H_e, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn,
                     T=12, n_episodes=300, alpha=0.05, gamma=0.90, epsilon=0.30,
                     budget=2000.0, seed=123):
    rng = np.random.default_rng(seed)
    Q = {}
    episode_returns = []
    available_paths = df_info_dyn["path"].unique()

    progress = st.progress(0.0, text="Entrenando Q-learning...")
    for ep in range(n_episodes):
        path = int(rng.choice(available_paths))
        state = initialize_state(nodes, edges, budget)
        total_return = 0.0

        for t in range(1, T + 1):
            W = get_exogenous_info(path, t, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn)
            s = discretize_state(state)
            if rng.random() < epsilon:
                a_name = rng.choice(ACTION_NAMES)
            else:
                values = [Q.get((s, a), 0.0) for a in ACTION_NAMES]
                a_name = ACTION_NAMES[int(np.argmin(values))]
            action = build_action(a_name, state, W, nodes, edges, H_e)
            next_state, info = transition(state, action, W, nodes, edges, H_e)
            cost = info["cost_breakdown"]["total_cost"]
            total_return += cost
            s_next = discretize_state(next_state)
            best_next = min(Q.get((s_next, a_next), 0.0) for a_next in ACTION_NAMES)
            old_q = Q.get((s, a_name), 0.0)
            Q[(s, a_name)] = old_q + alpha * (cost + gamma * best_next - old_q)
            state = next_state

        episode_returns.append(total_return)
        progress.progress((ep + 1) / n_episodes, text=f"Entrenando Q-learning... {ep+1}/{n_episodes}")
    progress.empty()
    return Q, episode_returns

# ============================================================
# Visualización
# ============================================================
def plot_network(edges, state=None):
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(int(r["i"]), int(r["j"]), e_id=int(r["e_id"]))

    pos = nx.spring_layout(G, seed=7, k=1 / max(len(G.nodes), 1) ** 0.5)
    edge_x, edge_y = [], []
    failed_x, failed_y = [], []
    reinforced_x, reinforced_y = [], []

    failed = state.get("failed", {}) if state else {}
    reinforced = state.get("reinforced", {}) if state else {}

    for u, v, data in G.edges(data=True):
        e = data["e_id"]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        target_x, target_y = edge_x, edge_y
        if failed.get(e, 0) == 1:
            target_x, target_y = failed_x, failed_y
        elif reinforced.get(e, 0) == 1:
            target_x, target_y = reinforced_x, reinforced_y
        target_x += [x0, x1, None]
        target_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#CBD5E1"), name="Normal"))
    fig.add_trace(go.Scatter(x=failed_x, y=failed_y, mode="lines", line=dict(width=2, color="#EF4444"), name="Fallado"))
    fig.add_trace(go.Scatter(x=reinforced_x, y=reinforced_y, mode="lines", line=dict(width=2, color="#22C55E"), name="Reforzado"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=6, color="#2563EB"), name="Nodos"))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0), showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def make_policy_df(Q):
    rows = []
    for s in sorted(set(k[0] for k in Q.keys())):
        values = {a: Q.get((s, a), 0.0) for a in ACTION_NAMES}
        best = min(values, key=values.get)
        rows.append({
            "crisis": s[0], "failed_bin": s[1], "reinforced_bin": s[2], "budget_bin": s[3], "deficit_bin": s[4],
            "best_action": best, **values
        })
    return pd.DataFrame(rows)


def make_policy_heatmap(policy_df, crisis_value=0, budget_bin=2, reinforced_bin=0):
    action_to_num = {a: i for i, a in enumerate(ACTION_NAMES)}
    num_to_action = {i: a for a, i in action_to_num.items()}
    df = policy_df.copy()
    df["action_num"] = df["best_action"].map(action_to_num)
    subset = df[
        (df["crisis"] == crisis_value) &
        (df["budget_bin"] == budget_bin) &
        (df["reinforced_bin"] == reinforced_bin)
    ]
    pivot = subset.pivot_table(
        index="failed_bin",
        columns="deficit_bin",
        values="action_num",
        aggfunc="first"
    ).reindex(index=[0, 1, 2, 3, 4, 5], columns=[0, 1, 2])
    z = pivot.values.astype(float)
    text = np.empty_like(z, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            text[i, j] = "No visitado" if np.isnan(z[i, j]) else num_to_action[int(z[i, j])]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        text=text,
        hovertemplate="Fallas bin=%{y}<br>Déficit bin=%{x}<br>Acción=%{text}<extra></extra>",
        colorscale="Viridis",
        zmin=0,
        zmax=max(len(ACTION_NAMES)-1, 1),
        colorbar=dict(
            title="Acción",
            tickmode="array",
            tickvals=list(action_to_num.values()),
            ticktext=list(action_to_num.keys())
        )
    ))
    fig.update_layout(
        title=f"Heatmap de política aprendida | crisis={crisis_value}, budget_bin={budget_bin}, reinforced_bin={reinforced_bin}",
        xaxis_title="Déficit acumulado discretizado",
        yaxis_title="Arcos fallados discretizados",
        height=520,
    )
    return fig

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("⚡ Laboratory")
page = st.sidebar.radio("Sección", ["🌎 World", "📐 Architecture", "🧠 Policy Evaluation", "🏆 Tournament"])
st.sidebar.divider()
st.sidebar.header("🌎 World Parameters")
T = st.sidebar.slider("T (horizon)", 4, 48, 12, 1)
seed = st.sidebar.number_input("Seed", value=2026, step=1)
n_paths = st.sidebar.slider("Paths", 10, 300, 80, 10)
p_crisis = st.sidebar.slider("P(crisis)", 0.00, 0.50, 0.15, 0.01)
dem_sigma = st.sidebar.slider("Demand noise σ", 0.0, 1.0, 0.40, 0.01)
cost_sigma = st.sidebar.slider("Cost noise σ", 0.0, 0.20, 0.05, 0.005)
budget = st.sidebar.number_input("Budget", value=2000.0, step=100.0)
st.sidebar.divider()
st.sidebar.header("🧠 Policy Settings")
n_episodes = st.sidebar.slider("Episodes Q-learning", 50, 2000, 300, 50)
alpha = st.sidebar.slider("α learning rate", 0.01, 0.50, 0.05, 0.01)
gamma = st.sidebar.slider("γ discount", 0.50, 0.99, 0.90, 0.01)
epsilon = st.sidebar.slider("ε exploration", 0.00, 0.80, 0.30, 0.01)

# ============================================================
# Cargar datos
# ============================================================
@st.cache_data(show_spinner=False)
def load_instance(instance=69):
    if load_edges_from_csv is None or load_nodes_from_csv is None:
        return None, None, "No se pudo importar scenarios_utils.py."
    if not (os.path.exists("edges.csv") and os.path.exists("nodes.csv")):
        return None, None, "No encontré edges.csv y nodes.csv en la carpeta del dashboard."
    edges_ = load_edges_from_csv("edges.csv")
    nodes_all_ = load_nodes_from_csv("nodes.csv")
    nodes_ = nodes_all_[nodes_all_["instance"] == instance].reset_index(drop=True)
    return nodes_, edges_, None

nodes, edges, data_error = load_instance(69)
if data_error:
    st.error(data_error)
    st.stop()

H_e = {int(e): 50.0 for e in edges["e_id"]}

@st.cache_data(show_spinner=False)
def cached_world(T, n_paths, seed, p_crisis, dem_sigma, cost_sigma):
    return generate_dynamic_scenarios(nodes, edges, T=T, n_paths=n_paths, seed=int(seed), p_crisis=p_crisis, dem_sigma=dem_sigma, cost_sigma=cost_sigma)

dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn = cached_world(T, n_paths, seed, p_crisis, dem_sigma, cost_sigma)

# ============================================================
# Pages
# ============================================================
if page == "🌎 World":
    st.title("🌎 Exogenous Information — W")
    st.write("El mundo genera demanda, capacidad, costos, generación y fallas en cada periodo. Ajusta los parámetros en la barra lateral para ver cómo cambia el proceso estocástico.")
    st.latex(r"W_{t+1}=(D_{t+1}, U_{t+1}, c_{t+1}, G_{t+1}, \xi_{t+1})")

    path0 = 0
    D_ts = dfD_dyn[dfD_dyn["path"] == path0].groupby("t")["D_it"].sum().reset_index(name="Demanda total")
    G_ts = dfG_dyn[dfG_dyn["path"] == path0].groupby("t")["G_it"].sum().reset_index(name="Generación disponible")
    U_ts = dfU_dyn[dfU_dyn["path"] == path0].groupby("t")["U_et"].sum().reset_index(name="Capacidad total")
    failed_ts = dfU_dyn[(dfU_dyn["path"] == path0)].groupby("t")["failed"].sum().reset_index(name="Arcos fallados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodos", len(nodes))
    c2.metric("Arcos", len(edges))
    c3.metric("Generadores", int(nodes.get("is_generator", pd.Series(False)).sum()))
    c4.metric("Periodos en crisis", int(df_info_dyn[df_info_dyn["path"] == path0]["crisis"].sum()))

    st.plotly_chart(px.line(D_ts, x="t", y="Demanda total", title="Demanda total por periodo"), use_container_width=True)
    st.plotly_chart(px.line(G_ts, x="t", y="Generación disponible", title="Generación disponible por periodo"), use_container_width=True)
    st.plotly_chart(px.line(U_ts, x="t", y="Capacidad total", title="Capacidad total de transmisión por periodo"), use_container_width=True)
    st.plotly_chart(px.bar(failed_ts, x="t", y="Arcos fallados", title="Arcos fallados por periodo"), use_container_width=True)

elif page == "📐 Architecture":
    st.title("📐 Mathematical Formulation")
    st.header("1 · Narrativa del problema")
    st.write("""
    Se modela una red eléctrica bajo incertidumbre dinámica. En cada periodo, el operador observa el estado de la red y decide cómo intervenir la infraestructura y cómo operar el sistema. La incertidumbre entra mediante demanda, generación disponible, costos, capacidades efectivas y fallas de arcos.
    """)
    st.header("2 · Cinco elementos de Powell")
    st.markdown("""
| Elemento | Símbolo | Definición en el modelo |
|---|---|---|
| Estado | $S_t$ | Capacidades, arcos fallados, arcos reforzados, presupuesto, demanda, generación, flujos, déficit y crisis. |
| Decisión | $x_t$ | Refuerzo, reparación, flujo por arco y generación despachada. |
| Información exógena | $W_{t+1}$ | Demanda, generación disponible, costos, capacidades efectivas, fallas y crisis. |
| Transición | $S^M$ | Actualiza capacidades, fallas, presupuesto, déficit y costo acumulado. |
| Costo | $C_t$ | Costos de operación, déficit, refuerzo y reparación. |
""")
    st.header("3 · Decisiones")
    st.latex(r"x_t=(h_{e,t}, r_{e,t}, f_{e,t}, g_{i,t})")
    st.markdown("""
- $h_{e,t}$: reforzar arco.
- $r_{e,t}$: reparar arco.
- $f_{e,t}$: flujo por arco.
- $g_{i,t}$: generación despachada por nodo.
""")
    st.header("4 · Transición")
    st.latex(r"S_{t+1}=S^M(S_t,x_t,W_{t+1})")
    st.latex(r"U_{e,t+1}=\begin{cases}0, & \xi_{e,t+1}=1 \\ \tilde U_{e,t+1}(1+\Delta U_e h_{e,t}), & \xi_{e,t+1}=0\end{cases}")
    st.header("5 · Costo por periodo")
    st.latex(r"C_t=\sum_{e\in E} c_{e,t}f_{e,t}+\pi\sum_{i\in N}\delta_{i,t}+\sum_{e\in E}H_eh_{e,t}+\sum_{e\in E}R_er_{e,t}")

    st.header("6 · Discretización del estado para Q-learning")
    st.markdown("""
Para poder aplicar Q-learning tabular, el estado continuo de la red se transforma en una representación discreta. El estado original contiene capacidades, fallas, presupuesto, flujos, generación y déficit, por lo que sería demasiado grande para almacenar una tabla Q(s,a). Por esta razón, se resume como:
""")
    st.latex(r"\bar S_t=(\text{crisis}_t,\text{failed\_bin}_t,\text{reinforced\_bin}_t,\text{budget\_bin}_t,\text{deficit\_bin}_t)")
    st.markdown("""
| Componente | Implementación | Interpretación |
|---|---|---|
| `crisis` | `int(state["crisis"])` | 0 para Mediocristan y 1 para Extremistan. |
| `failed_bin` | `min(failed // 10, 5)` | Agrupa el número de arcos fallados en bloques de 10. |
| `reinforced_bin` | `min(reinforced // 5, 5)` | Agrupa el número de arcos reforzados en bloques de 5. |
| `budget_bin` | bajo, medio, alto | Clasifica el presupuesto restante en 0, 1 o 2. |
| `deficit_bin` | nulo, moderado, alto | Resume el déficit acumulado del sistema. |
""")
    st.info("Esta discretización reduce la complejidad computacional, pero sacrifica detalle espacial: el agente sabe cuántos arcos fallaron, pero no exactamente cuáles.")

    st.header("7 · Acciones implementadas")
    st.markdown("""
La política no decide una única variable, sino una acción compuesta con cuatro componentes:
""")
    st.latex(r"x_t=(h_{e,t},r_{e,t},f_{e,t},g_{i,t})")
    st.markdown("""
| Decisión | Código | Tipo | Descripción |
|---|---|---|---|
| Reforzar arcos | `action["reinforce"]` | Binaria | Decide qué arcos aumentan su capacidad. |
| Reparar arcos | `action["repair"]` | Binaria | Decide qué arcos fallados se recuperan. |
| Flujo por arco | `action["flow"]` | Continua | Define energía enviada por cada arco. |
| Generación despachada | `action["generation_dispatch"]` | Continua | Define generación usada en cada nodo. |
""")

    st.subheader("Implementación de flujo y generación")
    st.markdown("""
Para mantener el espacio de acciones tratable, Q-learning aprende acciones agregadas (`do_nothing`, `repair_failed`, `reinforce_low_capacity`, `repair_and_reinforce`). Cada acción agregada se traduce internamente en las cuatro decisiones. En particular, la generación y el flujo se implementan con reglas simples:
""")
    st.code("""# Generación despachada: usa toda la generación disponible
for i in nodes_list:
    action["generation_dispatch"][i] = W["G"].get(i, 0.0)

# Flujo simple: permite flujo proporcional a capacidad disponible
# La transición luego valida capacidad y no negatividad.
for e in edges_list:
    if state["failed"].get(e, 0) == 0:
        action["flow"][e] = 0.25 * state["capacity"].get(e, 0.0)""", language="python")
    st.markdown("""
Esto significa que la generación despachada se iguala a la generación disponible observada en el periodo y que el flujo se aproxima como una proporción de la capacidad disponible en arcos no fallados. Luego, la transición valida que el flujo no sea negativo, que no supere la capacidad y que la generación no exceda la disponibilidad.
""")

elif page == "🧠 Policy Evaluation":
    st.title("🧠 Policy Evaluation — Q-learning")
    st.write("Entrena una política tabular Q-learning y evalúa un episodio fuera de muestra.")
    if st.button("🏁 Train Q-learning"):
        Q, q_returns = q_learning_train(nodes, edges, H_e, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn,
                                        T=T, n_episodes=n_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, budget=budget, seed=int(seed))
        st.session_state["Q"] = Q
        st.session_state["q_returns"] = q_returns

    if "Q" in st.session_state:
        Q = st.session_state["Q"]
        q_returns = st.session_state["q_returns"]
        returns_df = pd.DataFrame({"episode": np.arange(1, len(q_returns)+1), "cost": q_returns})
        returns_df["moving_avg"] = returns_df["cost"].rolling(30, min_periods=1).mean()
        st.plotly_chart(px.line(returns_df, x="episode", y=["cost", "moving_avg"], title="Convergencia Q-learning"), use_container_width=True)

        path_eval = st.slider("Path para evaluar", 0, int(df_info_dyn["path"].max()), 0, 1)
        ep = run_episode(q_learning_policy_fn, path_eval, T, nodes, edges, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn, H_e,
                         budget=budget, policy_params={"Q": Q})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Costo total", f"{ep['period_cost'].sum():,.0f}")
        k2.metric("Déficit total", f"{ep['total_deficit'].sum():,.1f}")
        k3.metric("Presupuesto final", f"{ep['budget_remaining'].iloc[-1]:,.0f}")
        k4.metric("Arcos reforzados", int(ep['num_reinforced_edges'].iloc[-1]))
        st.dataframe(ep, use_container_width=True)
        st.plotly_chart(px.line(ep, x="t", y=["period_cost", "total_deficit", "num_failed_edges"], title="Episodio evaluado"), use_container_width=True)

        policy_df = make_policy_df(Q)
        st.subheader("Política aprendida")
        st.dataframe(policy_df, use_container_width=True)

        st.subheader("Heatmap de política aprendida")
        st.write("El heatmap muestra qué acción escoge Q-learning para combinaciones de fallas discretizadas y déficit acumulado discretizado. Las dimensiones restantes del estado se fijan con los controles.")
        hm1, hm2, hm3 = st.columns(3)
        crisis_label = hm1.selectbox("Régimen", ["Mediocristan", "Extremistan"], index=0)
        crisis_value = 0 if crisis_label == "Mediocristan" else 1
        budget_bin_hm = hm2.selectbox("budget_bin", [0, 1, 2], index=2, help="0=bajo, 1=medio, 2=alto")
        reinforced_bin_hm = hm3.selectbox("reinforced_bin", [0, 1, 2, 3, 4, 5], index=0)
        fig_hm = make_policy_heatmap(policy_df, crisis_value=crisis_value, budget_bin=budget_bin_hm, reinforced_bin=reinforced_bin_hm)
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Entrena Q-learning para ver la política y los resultados.")

elif page == "🏆 Tournament":
    st.title("🏆 Tournament")
    st.write("Compara la política base contra Q-learning en varios paths.") 
    if "Q" not in st.session_state:
        st.warning("Primero entrena Q-learning en Policy Evaluation.") 
        st.stop()
    Q = st.session_state["Q"]
    n_eval = st.slider("Paths de evaluación", 5, min(50, n_paths), 20, 5)
    rows = []
    for p in range(n_eval):
        ep_base = run_episode(no_action_policy, p, T, nodes, edges, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn, H_e, budget=budget)
        ep_q = run_episode(q_learning_policy_fn, p, T, nodes, edges, dfD_dyn, dfU_dyn, dfc_dyn, dfG_dyn, df_info_dyn, H_e, budget=budget, policy_params={"Q": Q})
        rows.append({"policy": "No action", "path": p, "total_cost": ep_base["period_cost"].sum(), "total_deficit": ep_base["total_deficit"].sum()})
        rows.append({"policy": "Q-learning", "path": p, "total_cost": ep_q["period_cost"].sum(), "total_deficit": ep_q["total_deficit"].sum()})
    tournament_df = pd.DataFrame(rows)
    st.dataframe(tournament_df.groupby("policy").agg(avg_cost=("total_cost","mean"), avg_deficit=("total_deficit","mean")).reset_index(), use_container_width=True)
    st.plotly_chart(px.box(tournament_df, x="policy", y="total_cost", title="Distribución de costo por política"), use_container_width=True)
    st.plotly_chart(px.box(tournament_df, x="policy", y="total_deficit", title="Distribución de déficit por política"), use_container_width=True)
