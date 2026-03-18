#!/usr/bin/env python3

# ============================================================
# USER CONFIG (CHANGE ONLY HERE)
# ============================================================

# 1) Map set folder name: expects maps_<MAP_SET>/index.csv
#    Examples: "G5C","G5R","G5CUnk", "G8C", "G8R","G8CObs","G8RObs","G8CUnk"
MAP_SET = "G8C"

# 2) MapID range
# MAP_RANGE = (1,20)
# MAP_RANGE = (21,40)
# MAP_RANGE = (41,60)
# MAP_RANGE = (61,80)
# MAP_RANGE = (81,100)

# 3) Provider: "gemini" or "openai"
PROVIDER = "gemini"
# PROVIDER = "openai"

# 4) Model name
# Gemini examples: "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"
# OpenAI examples: "gpt-5.2", "gpt-4.1"
MODEL_NAME = "gemini-3-flash-preview"

# 5) Sleep between calls (seconds)
SLEEP_BETWEEN = 10.0

# 6) API KEYS (DIRECT INPUT)
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
# ============================================================

import os, time, re, csv, random, json
from datetime import datetime
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

# -------------------------
# Optional imports by provider
# -------------------------
def _init_gemini_client():
    from google import genai
    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key or not key.startswith("AIza"):
        raise RuntimeError("Invalid GEMINI_API_KEY. Set it in USER CONFIG or env GEMINI_API_KEY.")
    return genai.Client(api_key=key)

def _init_openai_client():
    from openai import OpenAI
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not key or not key.startswith("sk-"):
        raise RuntimeError("Invalid OPENAI_API_KEY. Set it in USER CONFIG or env OPENAI_API_KEY.")
    return OpenAI(api_key=key)

# =========================
# Prompt + extractors
# =========================
BASE_PROMPT_TEMPLATE = """You are a robot path planner.
Below is an ASCII grid map where:
- 'S' = Start point
- 'G' = Goal point
- '#' = Obstacles
- '?' = Unknown terrain
- '.' = Free space

Task:
1) Determine a safe path from S to G, avoiding obstacles (#).
   - You may move ONLY in four directions: up, down, left, and right.
   - Diagonal moves are NOT allowed.
   - Unknown (?) may or may not be passable; choose any reasonable assumption.
2) Draw the route directly ON THE MAP by replacing traversed '.' (and '?' if you choose to enter unknown) with '*'.
3) Also list the ordered coordinates, zero-indexed as (row,col), from S to G.

Map:
{MAP_TEXT}
"""

OUTPUT_FORMAT_INSTRUCTION = """
Output format (MANDATORY):

### Assumptions
I assume that unknown terrain (?) is not passable.
### Map with Path
<ASCII map with '*' drawn along the route>
### Coordinates
coords: (r,c)->(r,c)->...    # start from the first step after S; include G
"""

def extract_assumptions(text: str) -> str:
    m = re.search(r"###\s*Assumptions\s*(.*?)(?:\n\s*###|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""

def extract_map_block(text: str) -> str:
    m = re.search(r"###\s*Map with Path\s*```(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()

    m = re.search(r"###\s*Map with Path\s*(.*?)(?:\n\s*###|\Z)", text, re.S | re.I)
    if m:
        block = m.group(1).strip()
        block = re.sub(r"^`+|`+$", "", block).strip()
        return block

    m = re.search(r"(?:###\s*Updated map|Updated map)\s*```(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    return ""

def extract_coords_line(text: str) -> str:
    m = re.search(r"^\s*coords:\s*([^\n]+)", text, re.I | re.M)
    if m:
        line = "coords: " + m.group(1).strip()
        line = re.sub(r"\)\s*-\>\s*\(", ")->(", line)
        line = "coords: " + re.sub(r"\s+", "", line.replace("coords:", "")).strip()
        return line

    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if pairs:
        pairs_fmt = [f"({r},{c})" for (r, c) in pairs[1:]] if len(pairs) >= 2 else []
        if pairs_fmt:
            return "coords: " + "->".join(pairs_fmt)

    return "coords: (not found)"

# =========================
# Rate-limit + retry (Gemini)
# =========================
RPM_LIMIT = 8
RPM_SAFETY_MARGIN = 1.0
MAX_RETRIES = 6
BASE_BACKOFF = 1.5
BACKOFF_MULTIPLIER = 1.8
JITTER_RANGE = (0.2, 0.6)
_recent_calls = deque()

def _rate_limit_sleep_if_needed():
    now = time.time()
    while _recent_calls and now - _recent_calls[0] > 60:
        _recent_calls.popleft()
    if len(_recent_calls) >= RPM_LIMIT:
        wait_sec = 60 - (now - _recent_calls[0]) + RPM_SAFETY_MARGIN
        if wait_sec > 0:
            print(f"[rate-limit] Sleeping {wait_sec:.2f}s…")
            time.sleep(wait_sec)

def _record_call():
    _recent_calls.append(time.time())

def _parse_retry_after_seconds_from_error(err_text: str):
    marker = "Please retry in "
    if marker in err_text:
        sub = err_text.split(marker, 1)[1]
        num = ""
        for ch in sub:
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        if num:
            try:
                return int(float(num))
            except:
                pass

    marker2 = "retryDelay': '"
    if marker2 in err_text:
        sub = err_text.split(marker2, 1)[1]
        sec_str = sub.split("'", 1)[0]
        if sec_str.endswith("s"):
            sec_str = sec_str[:-1]
        try:
            return int(float(sec_str))
        except:
            pass

    try:
        jstart = err_text.find("{"); jend = err_text.rfind("}")
        if jstart != -1 and jend != -1 and jend > jstart:
            data = json.loads(err_text[jstart:jend+1])
            details = data.get("error", {}).get("details", [])
            for d in details:
                if d.get("@type", "").endswith("RetryInfo"):
                    rd = d.get("retryDelay")
                    if isinstance(rd, str) and rd.endswith("s"):
                        return int(float(rd[:-1]))
                    if isinstance(rd, (int, float)):
                        return int(rd)
    except:
        pass
    return None

def call_gemini(client, prompt: str, model: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _rate_limit_sleep_if_needed()
            resp = client.models.generate_content(model=model, contents=prompt)
            _record_call()
            return getattr(resp, "text", "").strip() if resp else ""
        except Exception as e:
            err_text = str(e)
            if "API_KEY_INVALID" in err_text or "API key not valid" in err_text:
                raise RuntimeError("Invalid Gemini API key. Check GEMINI_API_KEY.") from e

            last_err = e
            retry_after = _parse_retry_after_seconds_from_error(err_text)
            if retry_after is not None:
                wait_s = max(1, int(retry_after))
                print(f"[retryDelay] Sleeping {wait_s}s (server hint)")
                time.sleep(wait_s)
                continue

            backoff = BASE_BACKOFF * (BACKOFF_MULTIPLIER ** attempt)
            jitter = random.uniform(*JITTER_RANGE)
            wait_s = backoff + jitter
            print(f"[error] {e} -> backoff {wait_s:.2f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait_s)

    raise last_err

def call_openai(client, prompt: str, model: str) -> str:
    # minimal; you can add response_format etc later
    resp = client.responses.create(model=model, input=prompt)
    return getattr(resp, "output_text", "").strip()

# =========================
# Helpers
# =========================
def validate_range(rng):
    if not (isinstance(rng, tuple) and len(rng) == 2):
        raise ValueError("MAP_RANGE must be a tuple like (1,20).")
    a, b = int(rng[0]), int(rng[1])
    if a < 1 or b < a:
        raise ValueError("Invalid MAP_RANGE. Use like (1,20), (21,40), etc.")
    return a, b

def load_index(map_dir: str):
    index_csv = os.path.join(map_dir, "index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"index.csv not found: {index_csv}")
    with open(index_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f)), index_csv

def parse_start_goal(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    """
    Supports two index.csv styles:

    (A) Newer style:
      S_row, S_col, G_row, G_col

    (B) Older style:
      Start="(r,c)", Goal="(r,c)"
    """
    if row.get("S_row") and row.get("S_col") and row.get("G_row") and row.get("G_col"):
        return row["S_row"], row["S_col"], row["G_row"], row["G_col"]

    start = row.get("Start", "")
    goal = row.get("Goal", "")
    # extract ints from "(r,c)"
    sm = re.findall(r"-?\d+", start)
    gm = re.findall(r"-?\d+", goal)
    if len(sm) >= 2 and len(gm) >= 2:
        return sm[0], sm[1], gm[0], gm[1]

    return "", "", "", ""

# =========================
# Main
# =========================
def main():
    r_start, r_end = validate_range(MAP_RANGE)

    if PROVIDER not in ("gemini", "openai"):
        raise ValueError("PROVIDER must be 'gemini' or 'openai'.")

    map_dir = f"maps_{MAP_SET}"
    if not os.path.isdir(map_dir):
        raise FileNotFoundError(f"Map directory not found: {map_dir}  (expected 'maps_{MAP_SET}')")

    rows, index_csv = load_index(map_dir)

    # Filter by MapID
    selected = []
    for row in rows:
        try:
            mid = int(row.get("MapID", "0") or "0")
        except:
            continue
        if r_start <= mid <= r_end:
            selected.append(row)

    if not selected:
        raise RuntimeError(f"No maps found for {MAP_SET} in range {r_start}-{r_end} (check index.csv).")

    # Output naming (includes timestamp to avoid overwriting)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME)
    out_txt = f"{PROVIDER}_{safe_model}_{MAP_SET}_{r_start:03d}-{r_end:03d}_{ts}.txt"
    out_csv = f"{PROVIDER}_{safe_model}_{MAP_SET}_{r_start:03d}-{r_end:03d}_{ts}.csv"

    # init client + caller
    if PROVIDER == "gemini":
        client = _init_gemini_client()
        caller = lambda prompt: call_gemini(client, prompt, MODEL_NAME)
    else:
        client = _init_openai_client()
        caller = lambda prompt: call_openai(client, prompt, MODEL_NAME)

    # write headers
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Batch Results\n")
        f.write(f"provider={PROVIDER}\nmodel={MODEL_NAME}\nset={MAP_SET}\nrange={r_start:03d}-{r_end:03d}\n")
        f.write(f"map_dir={map_dir}\nindex_csv={index_csv}\n\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "MapID","Filename",
            "Rows","Cols","S_row","S_col","G_row","G_col",
            "ShortestPathLen","ObstacleRatio","UnknownRatio",
            "Assumptions","Map_with_Path","Coordinates",
            "InputMap"
        ])

    # run
    for row in selected:
        map_id = int(row["MapID"])
        fname = row.get("Filename") or row.get("filename") or f"{MAP_SET}_{map_id:03d}.txt"
        fpath = os.path.join(map_dir, fname)

        if not os.path.exists(fpath):
            print(f"[skip] missing {fpath}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            map_text = f.read().strip()

        prompt = BASE_PROMPT_TEMPLATE.format(MAP_TEXT=map_text) + OUTPUT_FORMAT_INSTRUCTION

        try:
            raw = caller(prompt)

            assumptions = extract_assumptions(raw)
            map_with_path = extract_map_block(raw)
            coords = extract_coords_line(raw)

            print(f"[OK] {MAP_SET} Map {map_id:03d} ({fname})")

            # append txt
            with open(out_txt, "a", encoding="utf-8") as f:
                f.write(f"\n================= {MAP_SET} Map {map_id:03d} ({fname}) =================\n")
                f.write("INPUT MAP:\n")
                f.write(map_text + "\n\n")
                
                f.write("OUTPUT:\n")
                f.write("### Assumptions\n")
                f.write((assumptions if assumptions else "(not found)") + "\n\n")
                f.write("### Map with Path\n")
                f.write((map_with_path if map_with_path else "(not found)") + "\n\n")
                f.write("### Coordinates\n")
                f.write(coords + "\n")
                f.write("============================================================\n\n")

            # index metadata fields (support both formats)
            s_row, s_col, g_row, g_col = parse_start_goal(row)

            # append csv
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                assumptions_csv = assumptions.replace("\n", " ") if assumptions else "(not found)"
                map_with_path_csv = map_with_path.replace("\n", "\\n") if map_with_path else "(not found)"
                input_map_csv = map_text.replace("\n", "\\n")

                w.writerow([
                    map_id, fname,
                    row.get("Rows",""), row.get("Cols",""),
                    s_row, s_col, g_row, g_col,
                    row.get("ShortestPathLen",""),
                    row.get("ObstacleRatio",""),
                    row.get("UnknownRatio",""),
                    assumptions_csv, map_with_path_csv, coords,
                    input_map_csv
                ])

        except Exception as e:
            print(f"[ERR] {MAP_SET} Map {map_id:03d} ({fname}): {e}")

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone.\nSaved TXT: {out_txt}\nSaved CSV: {out_csv}")

if __name__ == "__main__":
    main()
