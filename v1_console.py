import heapq

ROWS, COLS = 8, 8
START, GOAL = (0, 0), (7, 7)
OBSTACLES = {(1, 2), (2, 2), (3, 2), (4, 4), (4, 5), (4, 6), (5, 4), (6, 4)}


def heuristic(cell):
    """Manhattan distance from a cell to the goal."""
    return abs(cell[0] - GOAL[0]) + abs(cell[1] - GOAL[1])


def gbfs():
    frontier = [(heuristic(START), 0, START)]   # (h value, tie-break, cell)
    explored = set()
    parent = {}
    counter = 0
    while frontier:
        h, _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        if current == GOAL:
            path = [GOAL]
            while path[-1] != START:
                path.append(parent[path[-1]])
            return list(reversed(path)), explored
        r, c = current
        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if not (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS):
                continue
            if nxt in OBSTACLES or nxt in explored or nxt in parent or nxt == START:
                continue
            parent[nxt] = current
            counter += 1
            heapq.heappush(frontier, (heuristic(nxt), counter, nxt))
    return None, explored


def show(path, explored):
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            cell = (r, c)
            if cell == START:
                row += "S "
            elif cell == GOAL:
                row += "G "
            elif cell in OBSTACLES:
                row += "# "
            elif path and cell in path:
                row += "* "
            elif cell in explored:
                row += "o "
            else:
                row += ". "
        print(row)


path, explored = gbfs()
show(path, explored)
print()
print("Explored nodes:", len(explored))
print("Path length   :", len(path) - 1, "moves")
print("Path          :", path)
