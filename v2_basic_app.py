import heapq

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PathPilot | GBFS Lab", layout="wide")

ROWS, COLS = 8, 8
START, GOAL = (0, 0), (7, 7)
OBSTACLES = {(1, 2), (2, 2), (3, 2), (4, 4), (4, 5), (4, 6), (5, 4), (6, 4)}


def heuristic(cell):
    return abs(cell[0] - GOAL[0]) + abs(cell[1] - GOAL[1])


def new_state():
    return {
        "frontier": [(heuristic(START), 0, START)],
        "explored": set(),
        "parent": {},
        "path": [],
        "finished": False,
        "step": 0,
        "sequence": 0,
        "current": None,
        "message": "Press Next step to begin.",
    }


def advance_one(state):
    """Explore exactly one cell."""
    if state["finished"]:
        return
    while state["frontier"] and state["frontier"][0][2] in state["explored"]:
        heapq.heappop(state["frontier"])
    if not state["frontier"]:
        state["finished"] = True
        state["message"] = "No route exists."
        return
    h_val, _, current = heapq.heappop(state["frontier"])
    state["current"] = current
    state["explored"].add(current)
    state["step"] += 1
    if current == GOAL:
        path = [GOAL]
        while path[-1] != START:
            path.append(state["parent"][path[-1]])
        state["path"] = list(reversed(path))
        state["finished"] = True
        state["message"] = f"Goal reached after exploring {len(state['explored'])} cells."
        return
    r, c = current
    for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if not (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS):
            continue
        if nxt in OBSTACLES or nxt in state["explored"]:
            continue
        if nxt in state["parent"] or nxt == START:
            continue
        state["parent"][nxt] = current
        state["sequence"] += 1
        heapq.heappush(state["frontier"], (heuristic(nxt), state["sequence"], nxt))
    state["message"] = f"Selected {current} because h(n) = {h_val}."


def make_grid(state):
    frontier = {item[2] for item in state["frontier"] if item[2] not in state["explored"]}
    colors = {"start": "#34d399", "goal": "#fb7185", "wall": "#273449",
              "current": "#c084fc", "path": "#38bdf8", "explored": "#fbbf24",
              "frontier": "#fb923c", "free": "#dbe4f0"}
    xs, ys, fills = [], [], []
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell == START:
                kind = "start"
            elif cell == GOAL:
                kind = "goal"
            elif cell in OBSTACLES:
                kind = "wall"
            elif cell in state["path"]:
                kind = "path"
            elif cell == state["current"]:
                kind = "current"
            elif cell in state["explored"]:
                kind = "explored"
            elif cell in frontier:
                kind = "frontier"
            else:
                kind = "free"
            xs.append(c)
            ys.append(ROWS - 1 - r)
            fills.append(colors[kind])
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(symbol="square", size=48, color=fills,
                    line=dict(color="#50627d", width=1)),
    ))
    fig.update_layout(height=550, showlegend=False, dragmode=False,
                      xaxis=dict(range=[-0.6, 7.6], dtick=1),
                      yaxis=dict(range=[-0.6, 7.6], dtick=1, scaleanchor="x"))
    return fig


# Keep the search in session state so it survives each button click
if "gbfs" not in st.session_state:
    st.session_state.gbfs = new_state()
state = st.session_state.gbfs

st.title("PathPilot: Greedy Best-First Search")
st.plotly_chart(make_grid(state), use_container_width=True)
st.write(f"**Step {state['step']}.** {state['message']}")

col1, col2 = st.columns(2)
if col1.button("Next step", disabled=state["finished"]):
    advance_one(state)
    st.rerun()
if col2.button("Reset"):
    st.session_state.gbfs = new_state()
    st.rerun()
