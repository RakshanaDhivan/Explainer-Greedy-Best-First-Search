import heapq
import math

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PathPilot | GBFS Lab", layout="wide")

ROWS, COLS = 8, 8
START, GOAL = (0, 0), (7, 7)

# NEW: several maze layouts to experiment with
MAZES = {
    "Balanced maze": {(1, 2), (2, 2), (3, 2), (4, 4), (4, 5), (4, 6), (5, 4), (6, 4)},
    "Vertical barrier": {(1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)},
    "Open field": set(),
    "Zigzag challenge": {
        (1, 1), (1, 2), (1, 3), (1, 5), (1, 6),
        (2, 5), (3, 1), (3, 2), (3, 3), (3, 5),
        (4, 5), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6),
        (6, 1),
    },
}


# NEW: the heuristic can be switched between Manhattan and Euclidean
def get_heuristic(cell, goal, name):
    dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
    if name == "Euclidean":
        return math.sqrt(dr * dr + dc * dc)
    return dr + dc


def new_state(maze_name, heuristic_name):
    return {
        "maze_name": maze_name,
        "heuristic_name": heuristic_name,
        "obstacles": set(MAZES[maze_name]),
        "frontier": [(get_heuristic(START, GOAL, heuristic_name), 0, START)],
        "explored": set(),
        "parent": {},
        "path": [],
        "finished": False,
        "found": False,
        "step": 0,
        "sequence": 0,
        "current": None,
        "message": "Ready to explore. Select **Next step** to begin.",
    }


def advance_one(state):
    if state["finished"]:
        return
    frontier = state["frontier"]
    while frontier and frontier[0][2] in state["explored"]:
        heapq.heappop(frontier)
    if not frontier:
        state["finished"] = True
        state["found"] = False
        state["message"] = "No route exists in this maze."
        return
    h_val, _, current = heapq.heappop(frontier)
    state["current"] = current
    state["explored"].add(current)
    state["step"] += 1
    if current == GOAL:
        path = [GOAL]
        while path[-1] != START:
            path.append(state["parent"][path[-1]])
        state["path"] = list(reversed(path))
        state["finished"] = True
        state["found"] = True
        state["message"] = f"Goal reached! GBFS selected {current} with h(n) = {h_val:.2f}."
        return
    r, c = current
    added = []
    for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if not (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS):
            continue
        if nxt in state["obstacles"] or nxt in state["explored"]:
            continue
        if nxt in state["parent"] or nxt == START:
            continue
        state["parent"][nxt] = current
        state["sequence"] += 1
        nh = get_heuristic(nxt, GOAL, state["heuristic_name"])
        heapq.heappush(frontier, (nh, state["sequence"], nxt))
        added.append((nxt, nh))
    # NEW: the decision message lists every cell added and its h value
    added_text = ", ".join(f"{cell} (h={h:.1f})" for cell, h in added) or "no new cells"
    state["message"] = (
        f"Selected **{current}** with **h(n) = {h_val:.2f}**. Added: {added_text}."
    )


def make_grid(state):
    frontier = {item[2] for item in state["frontier"] if item[2] not in state["explored"]}
    colors = {"Start": "#34d399", "Goal": "#fb7185", "Obstacle": "#273449",
              "Current": "#c084fc", "Final path": "#38bdf8", "Explored": "#fbbf24",
              "Frontier": "#fb923c", "Unvisited": "#dbe4f0"}
    xs, ys, fills, labels, hover = [], [], [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell == START:
                typ, label = "Start", "S"
            elif cell == GOAL:
                typ, label = "Goal", "G"
            elif cell in state["obstacles"]:
                typ, label = "Obstacle", "×"
            elif cell in state["path"]:
                typ, label = "Final path", "●"
            elif cell == state["current"]:
                typ, label = "Current", "▶"
            elif cell in state["explored"]:
                typ, label = "Explored", "•"
            elif cell in frontier:
                # NEW: frontier cells show their h value on the grid
                typ = "Frontier"
                label = f"{get_heuristic(cell, GOAL, state['heuristic_name']):.0f}"
            else:
                typ, label = "Unvisited", ""
            xs.append(c)
            ys.append(ROWS - 1 - r)
            fills.append(colors[typ])
            labels.append(label)
            hv = get_heuristic(cell, GOAL, state["heuristic_name"])
            hover.append(f"Cell: {cell}<br>State: {typ}<br>h(n): {hv:.2f}")
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(symbol="square", size=48, color=fills,
                    line=dict(color="#50627d", width=1)),
        text=labels, textposition="middle center",
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(height=550, showlegend=False, dragmode=False,
                      xaxis=dict(range=[-0.6, 7.6], dtick=1),
                      yaxis=dict(range=[-0.6, 7.6], dtick=1, scaleanchor="x"))
    return fig


# ---------------- session state ----------------
def reset_search():
    st.session_state.gbfs = new_state(st.session_state.maze_choice,
                                      st.session_state.heuristic_choice)


if "maze_choice" not in st.session_state:
    st.session_state.maze_choice = "Balanced maze"
if "heuristic_choice" not in st.session_state:
    st.session_state.heuristic_choice = "Manhattan"
if "gbfs" not in st.session_state:
    reset_search()
state = st.session_state.gbfs

# NEW: sidebar with the experiment controls
with st.sidebar:
    st.header("Experiment controls")
    st.selectbox("Maze layout", list(MAZES.keys()), key="maze_choice", on_change=reset_search)
    st.selectbox("Heuristic function", ["Manhattan", "Euclidean"],
                 key="heuristic_choice", on_change=reset_search)
    st.caption("Changing either setting resets the search.")

st.title("PathPilot: Greedy Best-First Search")

# NEW: metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Search step", state["step"])
m2.metric("Explored nodes", len(state["explored"]))
m3.metric("Frontier size", len({i[2] for i in state["frontier"] if i[2] not in state["explored"]}))
m4.metric("Path length", f"{len(state['path']) - 1} moves" if state["found"] else "—")

st.plotly_chart(make_grid(state), use_container_width=True)

# NEW: Run to goal button and a result / decision box
a, b, c = st.columns(3)
if a.button("Next step", type="primary", disabled=state["finished"]):
    advance_one(state)
    st.rerun()
if b.button("Run to goal", disabled=state["finished"]):
    safety = 0
    while not state["finished"] and safety < ROWS * COLS * 2:
        advance_one(state)
        safety += 1
    st.rerun()
if c.button("Reset"):
    reset_search()
    st.rerun()

if state["finished"] and state["found"]:
    st.success(f"Goal reached in {state['step']} explored nodes. Path length: {len(state['path']) - 1} moves.")
elif state["finished"]:
    st.error("No route was found in this maze.")
else:
    st.info(f"**Decision:** {state['message']}")
