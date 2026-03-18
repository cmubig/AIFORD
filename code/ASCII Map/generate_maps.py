#!/usr/bin/env python3

import os, csv, random
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

# ============================================================
# ======================= USER CONFIG =========================
# ============================================================

RNG_SEED = 42
N_MAPS_PER_SET = 100
MAX_TRIES_PER_MAP = 20000

OUT_ROOT = "."  # output base directory

# Set name convention suggestion:
#   G8-C-O12-15-U8-12-L10-18
#   G5-R-O12-15-U0
#
# Define as many sets as you want in SETS below.

SETS = [
    # -------------------------
    # Examples (edit / add)
    # -------------------------
    dict(
        name="G5C",
        rows=5, cols=5,
        sg_mode="corner",          # "corner" or "random"
        obs_min=0.12, obs_max=0.15,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=8, lsp_max=12
    ),
    dict(
        name="G5R",
        rows=5, cols=5,
        sg_mode="random",
        obs_min=0.12, obs_max=0.15,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=6, lsp_max=15
    ),
    dict(
        name="G8C",
        rows=8, cols=8,
        sg_mode="corner",
        obs_min=0.12, obs_max=0.15,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=10, lsp_max=18
    ),
    dict(
        name="G8R",
        rows=8, cols=8,
        sg_mode="random",
        obs_min=0.12, obs_max=0.15,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=10, lsp_max=22
    ),
    dict(
        name="G8CObs",
        rows=8, cols=8,
        sg_mode="corner",
        obs_min=0.18, obs_max=0.24,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=14, lsp_max=26
    ),
    dict(
        name="G8RObs",
        rows=8, cols=8,
        sg_mode="random",
        obs_min=0.18, obs_max=0.24,
        unk_min=0.0,  unk_max=0.0,
        lsp_min=10, lsp_max=28
    ),
    dict(
        name="G5CUnk",
        rows=5, cols=5,
        sg_mode="corner",
        obs_min=0.12, obs_max=0.15,
        unk_min=0.08,  unk_max=0.12,
        lsp_min=8, lsp_max=12
    ),
    dict(
        name="G8CUnk",
        rows=8, cols=8,
        sg_mode="corner",
        obs_min=0.12, obs_max=0.15,
        unk_min=0.08, unk_max=0.12,
        lsp_min=10, lsp_max=22
    ),
    
]

# ============================================================
# ===================== END USER CONFIG ======================
# ============================================================

Coord = Tuple[int, int]
Grid = List[List[str]]

# =========================
# BFS shortest path
# =========================
def bfs_shortest_path(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    R, C = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    def passable(r: int, c: int) -> bool:
        return grid[r][c] != "#"

    q = deque([(sr, sc)])
    parent = {(sr, sc): None}

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            path = []
            cur = (gr, gc)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in parent and passable(nr, nc):
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    return None

def grid_to_ascii(grid: Grid) -> str:
    return "\n".join(" ".join(row) for row in grid)

# =========================
# S/G selection
# =========================
def corner_sg(rows: int, cols: int) -> Tuple[Coord, Coord]:
    return (0, 0), (rows - 1, cols - 1)

def random_sg(rows: int, cols: int, rng: random.Random) -> Tuple[Coord, Coord]:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    min_md = max(3, (rows + cols) // 4)
    while True:
        s = rng.choice(all_cells)
        g = rng.choice(all_cells)
        if s == g:
            continue
        if abs(s[0]-g[0]) + abs(s[1]-g[1]) < min_md:
            continue
        return s, g

# =========================
# One-map generator
# =========================
def generate_one_map(
    rng: random.Random,
    rows: int,
    cols: int,
    sg_mode: str,
    obs_min: float,
    obs_max: float,
    unk_min: float,
    unk_max: float,
    lsp_min: Optional[int],
    lsp_max: Optional[int],
    max_tries: int,
) -> Tuple[str, Dict[str, Any]]:

    for _ in range(max_tries):
        # choose S/G
        if sg_mode == "corner":
            start, goal = corner_sg(rows, cols)
        elif sg_mode == "random":
            start, goal = random_sg(rows, cols, rng)
        else:
            raise ValueError("sg_mode must be 'corner' or 'random'")

        # init grid
        grid: Grid = [["." for _ in range(cols)] for _ in range(rows)]
        grid[start[0]][start[1]] = "S"
        grid[goal[0]][goal[1]] = "G"

        # candidate cells excluding S/G
        cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in (start, goal)]
        rng.shuffle(cells)

        # obstacles
        obs_ratio = rng.uniform(obs_min, obs_max)
        n_obs = int(round(obs_ratio * len(cells)))
        for (r, c) in cells[:n_obs]:
            grid[r][c] = "#"

        # unknown (not overlapping obstacles)
        unk_ratio = rng.uniform(unk_min, unk_max) if (unk_max > 0.0) else 0.0
        n_unk = int(round(unk_ratio * len(cells)))

        remaining = [(r, c) for (r, c) in cells[n_obs:] if grid[r][c] == "."]
        n_unk = min(n_unk, len(remaining))
        if n_unk > 0:
            for (r, c) in rng.sample(remaining, n_unk):
                grid[r][c] = "?"

        # feasibility + Lsp
        sp = bfs_shortest_path(grid, start, goal)
        if sp is None:
            continue

        lsp = len(sp) - 1  # edges

        if lsp_min is not None and lsp < lsp_min:
            continue
        if lsp_max is not None and lsp > lsp_max:
            continue

        ascii_map = grid_to_ascii(grid)
        meta = {
            "rows": rows, "cols": cols,
            "start": start, "goal": goal,
            "obstacle_ratio": obs_ratio,
            "unknown_ratio": unk_ratio,
            "shortest_path_len": lsp,
        }
        return ascii_map, meta

    raise RuntimeError("Failed to generate a valid map within max_tries. Relax constraints or increase max_tries.")

# =========================
# Generate one set
# =========================
def generate_set(set_cfg: Dict[str, Any], base_seed: int) -> None:
    name = set_cfg["name"]
    rows = int(set_cfg["rows"]); cols = int(set_cfg["cols"])
    sg_mode = str(set_cfg["sg_mode"])
    obs_min = float(set_cfg["obs_min"]); obs_max = float(set_cfg["obs_max"])
    unk_min = float(set_cfg["unk_min"]); unk_max = float(set_cfg["unk_max"])
    lsp_min = set_cfg.get("lsp_min", None)
    lsp_max = set_cfg.get("lsp_max", None)

    # sanity checks
    if not (0.0 <= obs_min <= obs_max <= 1.0):
        raise ValueError(f"[{name}] obstacle ratios must satisfy 0<=min<=max<=1")
    if not (0.0 <= unk_min <= unk_max <= 1.0):
        raise ValueError(f"[{name}] unknown ratios must satisfy 0<=min<=max<=1")
    if lsp_min is not None and lsp_min < 0:
        raise ValueError(f"[{name}] lsp_min must be >= 0")
    if lsp_max is not None and lsp_max < 0:
        raise ValueError(f"[{name}] lsp_max must be >= 0")
    if lsp_min is not None and lsp_max is not None and lsp_min > lsp_max:
        raise ValueError(f"[{name}] require lsp_min <= lsp_max")

    out_dir = os.path.join(OUT_ROOT, f"maps_{name}")
    os.makedirs(out_dir, exist_ok=True)
    index_csv = os.path.join(out_dir, "index.csv")

    # deterministic but distinct per set
    rng = random.Random(base_seed + (abs(hash(name)) % 100000))

    seen = set()
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "MapID", "SetName",
            "Rows", "Cols",
            "SG_Mode", "Start", "Goal",
            "ObstacleRatio", "UnknownRatio",
            "ShortestPathLen",
            "Filename",
        ])

        map_id = 1
        while map_id <= N_MAPS_PER_SET:
            ascii_map, meta = generate_one_map(
                rng=rng,
                rows=rows, cols=cols,
                sg_mode=sg_mode,
                obs_min=obs_min, obs_max=obs_max,
                unk_min=unk_min, unk_max=unk_max,
                lsp_min=lsp_min, lsp_max=lsp_max,
                max_tries=MAX_TRIES_PER_MAP,
            )

            if ascii_map in seen:
                continue
            seen.add(ascii_map)

            fname = f"{name}_{map_id:03d}.txt"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as mf:
                mf.write(ascii_map + "\n")

            w.writerow([
                map_id, name,
                rows, cols,
                sg_mode, str(meta["start"]), str(meta["goal"]),
                f"{meta['obstacle_ratio']:.4f}",
                f"{meta['unknown_ratio']:.4f}",
                meta["shortest_path_len"],
                fname
            ])

            print(
                f"[OK] {fname}  "
                f"S={meta['start']} G={meta['goal']}  "
                f"Lsp={meta['shortest_path_len']}  "
                f"obs={meta['obstacle_ratio']:.3f}  unk={meta['unknown_ratio']:.3f}"
            )
            map_id += 1

    print(f"\n[{name}] Generated {N_MAPS_PER_SET} maps in '{out_dir}'. index: '{index_csv}'\n")

# =========================
# Main
# =========================
def main():
    print("=== Generating map sets ===")
    for cfg in SETS:
        generate_set(cfg, base_seed=RNG_SEED)
    print("=== Done ===")

if __name__ == "__main__":
    main()
