"""
DLA solvers — HiGHS (highspy).

HiGHS es gratuito, sin límite de tamaño, y es el mismo motor interno de scipy.milp.

IMPORTANTE: highspy.addVar() retorna HighsStatus, NO el índice.
El índice correcto es h.getNumCol() ANTES de llamar addVar.

Funciones exportadas:
  - solve_dla_deterministic_optimization
  - dla_deterministic_optimization_policy_fn
  - solve_dla_stochastic_optimization
  - dla_stochastic_optimization_policy_fn
"""

import highspy
import numpy as np

_INF = highspy.kHighsInf
_OPT = highspy.HighsModelStatus.kOptimal


def _new_model(time_limit=30):
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("time_limit", float(time_limit))
    return h


def _add_var(h, lb=0.0, ub=None, cost=0.0, binary=False):
    """Agrega una variable y retorna su índice correcto."""
    idx = h.getNumCol()                        # índice ANTES de agregar
    h.addVar(lb, _INF if ub is None else ub)
    h.changeColCost(idx, cost)
    if binary:
        h.changeColIntegrality(idx, highspy.HighsVarType.kInteger)
    return idx


def _add_row(h, lb, ub, indices, coeffs):
    n = len(indices)
    h.addRow(float(lb), float(ub),
             n,
             np.array(indices, dtype=np.int32),
             np.array(coeffs,  dtype=np.float64))


def _has_solution(h):
    return h.getModelStatus() == _OPT


# ─────────────────────────────────────────────────────────────────────────────
# DLA DETERMINÍSTICO
# ─────────────────────────────────────────────────────────────────────────────

def solve_dla_deterministic_optimization(state, W_current, nodes, edges, H_e, params):
    """
    DLA determinístico con HiGHS.
    Resuelve un MILP sobre horizonte H con pronóstico puntual.
    Retorna (model, var_index, W_list).
    """
    W_list      = build_deterministic_W_list(state, W_current, params)
    H           = len(W_list)
    gamma       = params.get("gamma", 0.95)
    delta_u     = params.get("delta_u", 0.8)
    repair_cost = params.get("repair_cost", 25.0)
    pi          = params.get("pi", 190.0)
    max_edges   = params.get("max_edges", 3)
    time_limit  = params.get("time_limit", 30)

    nodes_list = nodes["node"].astype(int).tolist()
    edges_list = edges["e_id"].astype(int).tolist()
    in_arcs  = {i: edges.loc[edges["j"] == i, "e_id"].astype(int).tolist() for i in nodes_list}
    out_arcs = {i: edges.loc[edges["i"] == i, "e_id"].astype(int).tolist() for i in nodes_list}

    h = _new_model(time_limit)
    var_index = {}

    # ── Variables ─────────────────────────────────────────────────────────────
    for period, W in enumerate(W_list):
        disc = gamma ** period
        for e in edges_list:
            var_index[("f", period, e)]        = _add_var(h, lb=0.0, cost=disc * W["c"].get(e, 0.0))
            var_index[("reinforce", period, e)] = _add_var(
                h, lb=0.0, ub=0.0 if state["reinforced"].get(e, 0) == 1 else 1.0,
                cost=disc * H_e[e], binary=True
            )
            var_index[("repair", period, e)]    = _add_var(h, lb=0.0, ub=1.0, cost=disc * repair_cost, binary=True)

        for i in nodes_list:
            var_index[("g", period, i)]     = _add_var(h, lb=0.0, ub=W["G"].get(i, 0.0), cost=0.0)
            var_index[("delta", period, i)] = _add_var(h, lb=0.0, cost=disc * pi)

    # ── Balance nodal: g_i + inflow - outflow + delta_i >= D_i ───────────────
    for period, W in enumerate(W_list):
        for i in nodes_list:
            idxs   = [var_index[("g", period, i)], var_index[("delta", period, i)]]
            coeffs = [1.0, 1.0]
            for e in in_arcs.get(i, []):
                idxs.append(var_index[("f", period, e)]); coeffs.append(1.0)
            for e in out_arcs.get(i, []):
                idxs.append(var_index[("f", period, e)]); coeffs.append(-1.0)
            _add_row(h, W["D"].get(i, 0.0), _INF, idxs, coeffs)

    # ── Capacidad de flujo ────────────────────────────────────────────────────
    for period, W in enumerate(W_list):
        for e in edges_list:
            U_e                = max(float(W["U"].get(e, 0.0)), 0.0)
            already_reinforced = state["reinforced"].get(e, 0)
            initially_failed   = state["failed"].get(e, 0)
            exog_failed        = W["failed"].get(e, 0)
            f_idx              = var_index[("f", period, e)]

            if exog_failed == 1:
                h.changeColBounds(f_idx, 0.0, 0.0)
                h.changeColBounds(var_index[("repair", period, e)], 0.0, 0.0)

            elif initially_failed == 1:
                # reparaciones acumuladas restituyen capacidad
                idxs   = [f_idx]
                coeffs = [1.0]
                for k in range(period + 1):
                    idxs.append(var_index[("repair", k, e)]); coeffs.append(-U_e)
                _add_row(h, -_INF, 0.0, idxs, coeffs)

            else:
                base_cap = U_e * (1.0 + delta_u * already_reinforced)
                idxs   = [f_idx]
                coeffs = [1.0]
                if already_reinforced == 0:
                    for k in range(period + 1):
                        idxs.append(var_index[("reinforce", k, e)]); coeffs.append(-delta_u * U_e)
                _add_row(h, -_INF, base_cap, idxs, coeffs)

    # ── Máximo de intervenciones por periodo ──────────────────────────────────
    for period in range(H):
        _add_row(h, 0, max_edges,
                 [var_index[("reinforce", period, e)] for e in edges_list],
                 [1.0] * len(edges_list))
        _add_row(h, 0, max_edges,
                 [var_index[("repair", period, e)] for e in edges_list],
                 [1.0] * len(edges_list))

    # ── Presupuesto total ─────────────────────────────────────────────────────
    idxs, coeffs = [], []
    for period in range(H):
        for e in edges_list:
            idxs.append(var_index[("reinforce", period, e)]); coeffs.append(H_e[e])
            idxs.append(var_index[("repair",    period, e)]); coeffs.append(repair_cost)
    _add_row(h, 0, state["budget_remaining"], idxs, coeffs)

    h.run()
    return h, var_index, W_list


def dla_deterministic_optimization_policy_fn(state, W, nodes, edges, H_e, params=None):
    if params is None:
        params = {}

    m, var_index, W_list = solve_dla_deterministic_optimization(
        state=state, W_current=W, nodes=nodes, edges=edges, H_e=H_e, params=params
    )
    edges_list = edges["e_id"].astype(int).tolist()
    nodes_list = nodes["node"].astype(int).tolist()

    if not _has_solution(m):
        return build_action("do_nothing", state=state, W=W, nodes=nodes, edges=edges,
                            H_e=H_e, max_edges=params.get("max_edges", 3))

    x = m.getSolution().col_value
    action = {"reinforce": {}, "repair": {}, "flow": {}, "generation_dispatch": {}}
    for e in edges_list:
        action["reinforce"][e]  = int(round(x[var_index[("reinforce", 0, e)]]))
        action["repair"][e]     = int(round(x[var_index[("repair",    0, e)]]))
        action["flow"][e]       = max(float(x[var_index[("f",         0, e)]]), 0.0)
    for i in nodes_list:
        action["generation_dispatch"][i] = max(float(x[var_index[("g", 0, i)]]), 0.0)
    return action


# ─────────────────────────────────────────────────────────────────────────────
# DLA ESTOCÁSTICO
# ─────────────────────────────────────────────────────────────────────────────

def solve_dla_stochastic_optimization(state, W_current, nodes, edges, H_e, params):
    """
    DLA estocástico con HiGHS.
    S escenarios, H periodos, restricciones de no-anticipatividad en h=0.
    Retorna (model, var_index).
    """
    H           = params.get("H", 2)
    T           = params["T"]
    S           = params.get("num_scenarios", 5)
    gamma       = params.get("gamma", 0.95)
    delta_u     = params.get("delta_u", 0.8)
    repair_cost = params.get("repair_cost", 25.0)
    pi          = params.get("pi", 190.0)
    max_edges   = params.get("max_edges", 3)
    time_limit  = params.get("time_limit", 60)

    available_paths = sorted(params["df_info_dyn"]["path"].unique())[:S]
    S    = len(available_paths)
    prob = 1.0 / S

    nodes_list = nodes["node"].astype(int).tolist()
    edges_list = edges["e_id"].astype(int).tolist()
    in_arcs  = {i: edges.loc[edges["j"] == i, "e_id"].astype(int).tolist() for i in nodes_list}
    out_arcs = {i: edges.loc[edges["i"] == i, "e_id"].astype(int).tolist() for i in nodes_list}

    # ── Árbol de información ──────────────────────────────────────────────────
    W_tree = {}
    for s, path in enumerate(available_paths):
        W_tree[(s, 0)] = W_current
        for period in range(1, H):
            t_future = state["t"] + 1 + period
            if t_future <= T:
                W_tree[(s, period)] = get_exogenous_info(
                    path=path, t=t_future,
                    dfD_dyn=params["dfD_dyn"], dfU_dyn=params["dfU_dyn"],
                    dfc_dyn=params["dfc_dyn"], dfG_dyn=params["dfG_dyn"],
                    df_info_dyn=params["df_info_dyn"]
                )

    h = _new_model(time_limit)
    var_index = {}

    # ── Variables ─────────────────────────────────────────────────────────────
    for s in range(S):
        for period in range(H):
            if (s, period) not in W_tree:
                continue
            W    = W_tree[(s, period)]
            disc = prob * (gamma ** period)

            for e in edges_list:
                var_index[("f",         s, period, e)] = _add_var(h, lb=0.0, cost=disc * W["c"].get(e, 0.0))
                var_index[("reinforce", s, period, e)] = _add_var(
                    h, lb=0.0, ub=0.0 if state["reinforced"].get(e, 0) == 1 else 1.0,
                    cost=disc * H_e[e], binary=True
                )
                var_index[("repair",    s, period, e)] = _add_var(h, lb=0.0, ub=1.0, cost=disc * repair_cost, binary=True)

            for i in nodes_list:
                var_index[("g",     s, period, i)] = _add_var(h, lb=0.0, ub=W["G"].get(i, 0.0), cost=0.0)
                var_index[("delta", s, period, i)] = _add_var(h, lb=0.0, cost=disc * pi)

    # ── No-anticipatividad en h=0 ─────────────────────────────────────────────
    for s in range(1, S):
        for e in edges_list:
            for vtype in ("f", "reinforce", "repair"):
                _add_row(h, 0.0, 0.0,
                         [var_index[(vtype, s, 0, e)], var_index[(vtype, 0, 0, e)]],
                         [1.0, -1.0])
        for i in nodes_list:
            for vtype in ("g", "delta"):
                _add_row(h, 0.0, 0.0,
                         [var_index[(vtype, s, 0, i)], var_index[(vtype, 0, 0, i)]],
                         [1.0, -1.0])

    # ── Balance nodal ─────────────────────────────────────────────────────────
    for s in range(S):
        for period in range(H):
            if (s, period) not in W_tree:
                continue
            W = W_tree[(s, period)]
            for i in nodes_list:
                idxs   = [var_index[("g", s, period, i)], var_index[("delta", s, period, i)]]
                coeffs = [1.0, 1.0]
                for e in in_arcs.get(i, []):
                    idxs.append(var_index[("f", s, period, e)]); coeffs.append(1.0)
                for e in out_arcs.get(i, []):
                    idxs.append(var_index[("f", s, period, e)]); coeffs.append(-1.0)
                _add_row(h, W["D"].get(i, 0.0), _INF, idxs, coeffs)

    # ── Capacidad de flujo ────────────────────────────────────────────────────
    for s in range(S):
        for period in range(H):
            if (s, period) not in W_tree:
                continue
            W = W_tree[(s, period)]
            for e in edges_list:
                U_e                = max(float(W["U"].get(e, 0.0)), 0.0)
                already_reinforced = state["reinforced"].get(e, 0)
                initially_failed   = state["failed"].get(e, 0)
                exog_failed        = W["failed"].get(e, 0)
                f_idx              = var_index[("f", s, period, e)]

                if exog_failed == 1:
                    h.changeColBounds(f_idx, 0.0, 0.0)
                    h.changeColBounds(var_index[("repair", s, period, e)], 0.0, 0.0)

                elif initially_failed == 1:
                    idxs   = [f_idx]
                    coeffs = [1.0]
                    for k in range(period + 1):
                        if (s, k) in W_tree:
                            idxs.append(var_index[("repair", s, k, e)]); coeffs.append(-U_e)
                    _add_row(h, -_INF, 0.0, idxs, coeffs)

                else:
                    base_cap = U_e * (1.0 + delta_u * already_reinforced)
                    idxs   = [f_idx]
                    coeffs = [1.0]
                    if already_reinforced == 0:
                        for k in range(period + 1):
                            if (s, k) in W_tree:
                                idxs.append(var_index[("reinforce", s, k, e)]); coeffs.append(-delta_u * U_e)
                    _add_row(h, -_INF, base_cap, idxs, coeffs)

    # ── Máximo de intervenciones ──────────────────────────────────────────────
    for s in range(S):
        for period in range(H):
            if (s, period) not in W_tree:
                continue
            _add_row(h, 0, max_edges,
                     [var_index[("reinforce", s, period, e)] for e in edges_list],
                     [1.0] * len(edges_list))
            _add_row(h, 0, max_edges,
                     [var_index[("repair", s, period, e)] for e in edges_list],
                     [1.0] * len(edges_list))

    # ── Presupuesto por escenario ─────────────────────────────────────────────
    for s in range(S):
        idxs, coeffs = [], []
        for period in range(H):
            if (s, period) not in W_tree:
                continue
            for e in edges_list:
                idxs.append(var_index[("reinforce", s, period, e)]); coeffs.append(H_e[e])
                idxs.append(var_index[("repair",    s, period, e)]); coeffs.append(repair_cost)
        if idxs:
            _add_row(h, 0, state["budget_remaining"], idxs, coeffs)

    h.run()
    return h, var_index


def dla_stochastic_optimization_policy_fn(state, W, nodes, edges, H_e, params=None):
    if params is None:
        params = {}

    m, var_index = solve_dla_stochastic_optimization(
        state=state, W_current=W, nodes=nodes, edges=edges, H_e=H_e, params=params
    )
    edges_list = edges["e_id"].astype(int).tolist()
    nodes_list = nodes["node"].astype(int).tolist()

    if not _has_solution(m):
        return build_action("do_nothing", state=state, W=W, nodes=nodes, edges=edges,
                            H_e=H_e, max_edges=params.get("max_edges", 3))

    x = m.getSolution().col_value
    action = {"reinforce": {}, "repair": {}, "flow": {}, "generation_dispatch": {}}
    for e in edges_list:
        action["reinforce"][e]  = int(round(x[var_index[("reinforce", 0, 0, e)]]))
        action["repair"][e]     = int(round(x[var_index[("repair",    0, 0, e)]]))
        action["flow"][e]       = max(float(x[var_index[("f",         0, 0, e)]]), 0.0)
    for i in nodes_list:
        action["generation_dispatch"][i] = max(float(x[var_index[("g", 0, 0, i)]]), 0.0)
    return action
