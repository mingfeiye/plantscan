"""
agent_review.py
The agent step. It does the judgment that a fixed script cannot do.

1. Fixed rules find the DOUBTFUL cases in output/inventory.json:
   - an object much wider than the others, or much longer than it is wide
     (possibly two plants that touch),
   - an object at the edge of the scanned surface (possibly cut off),
   - a count that is different from the expected count (if you give one).
2. Claude (through the Claude Code command `claude -p`, with your Claude login) looks at
   the measurements and the top-view image and, for each doubtful case, DECIDES
   (count as one, count as two, not a plant) or FLAGS it for a person, with a reason in
   one sentence. Then it answers in plain language: how many plants at each grade.
3. The program checks the agent's numbers against its own count. If they differ, or if
   Claude is not available (no internet, timeout), the program still finishes: every
   doubtful case is flagged for a person and the counts come from the program itself.

Use (run in the project folder, after plant_inventory.py):
    venv/bin/python agent_review.py                 # no expected count
    venv/bin/python agent_review.py --expected 9    # you expect 9 plants
Output: output/agent.json
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from statistics import median

ACTIONS = ["count_as_one", "count_as_two", "not_a_plant", "flag_for_person"]
TIMEOUT_S = 40         # keeps the whole run under 1 minute even when the agent is slow
EDGE_CM = 3.0          # closer than this to the edge of the scanned surface: maybe cut off
WIDE_FACTOR = 1.6      # max width more than 1.6 x the typical spread: maybe two plants
LONG_FACTOR = 1.3      # max width more than 1.3 x its own spread: not round, maybe two plants


# ---------------------------------------------------------------- 1. fixed rules
def find_doubts(result, expected):
    objs = result["objects"]
    typical = median(o["spread_cm"] for o in objs) if len(objs) >= 3 else None
    doubts = []
    for o in objs:
        why = []
        if typical and o["max_width_cm"] > WIDE_FACTOR * typical:
            why.append(f"it is {o['max_width_cm']} cm wide, but a typical plant here is "
                       f"{typical:.0f} cm wide")
        if o["max_width_cm"] > LONG_FACTOR * o["spread_cm"] and o["max_width_cm"] > 8:
            why.append(f"it is not round (longest width {o['max_width_cm']} cm, "
                       f"average width {o['spread_cm']} cm)")
        if o["edge_distance_cm"] <= EDGE_CM:
            why.append("it touches the edge of the scanned surface, so part of it may be cut off")
        if o.get("split_from_touching"):
            why.append("the program split it from a group of touching plants")
        if why:
            doubts.append({"id": o["id"], "rule_reasons": why})
    count_doubt = None
    if expected is not None and expected != len(objs):
        count_doubt = f"The program found {len(objs)} objects, but {expected} were expected."
    return doubts, count_doubt


# ---------------------------------------------------------------- 2. ask Claude
SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"},
                           "action": {"type": "string", "enum": ACTIONS},
                           "reason": {"type": "string"}},
            "required": ["id", "action", "reason"]}},
        "count_note": {"type": "string"},
        "grade_counts": {"type": "array", "items": {
            "type": "object",
            "properties": {"grade": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["grade", "count"]}},
        "answer": {"type": "string"},
    },
    "required": ["decisions", "count_note", "grade_counts", "answer"],
}


def build_prompt(result, doubts, count_doubt, expected, image_path):
    table = [{k: o[k] for k in ("id", "height_cm", "spread_cm", "max_width_cm",
                                "edge_distance_cm", "split_from_touching", "grade")}
             for o in result["objects"]]
    grades = list(result["grade_counts"].keys())
    return f"""You check the result of a plant inventory made from a 3D phone scan of container
plants on a surface. A program already measured each object. Your job is the judgment that a
fixed program cannot make.

First, read the top-view image at {image_path} with the Read tool. Each object has a red outline
and a number. Colour shows height above the surface.

Measurements (cm). edge_distance_cm is the distance from the object to the edge of the scanned
surface; near 0 means the object may be partly outside the scan.
{json.dumps(table, indent=1)}

Doubtful cases found by fixed rules:
{json.dumps(doubts, indent=1) if doubts else "none"}
Count check: {count_doubt or ("the count matches the expected count" if expected else "no expected count was given")}

For EACH doubtful case (and any other object that looks wrong in the image), choose one action:
- count_as_one: it is one plant.
- count_as_two: it is two plants that touch and were measured as one.
- not_a_plant: it is not a plant (for example a piece of wall, furniture or a scan error).
- flag_for_person: you are not sure; a person must check it.
Give the reason in ONE plain sentence. Do not invent measurements. If you are not sure, flag it.
In count_note, say in one sentence whether the final count is trustworthy.
Then give grade_counts for the final count after your decisions, using exactly these grade names:
{json.dumps(grades)}
(an object counted as two gives two plants of its grade; not_a_plant gives zero; every other
object gives one). Finally, write "answer": two or three plain sentences for a nursery grower
that say how many plants there are at each grade and how many need a person to check."""


def ask_claude(prompt, log=print):
    exe = shutil.which("claude")
    if not exe:
        return None, "the claude command is not installed"
    cmd = [exe, "-p", prompt, "--model", "sonnet", "--tools", "Read", "--allowedTools", "Read",
           "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
           "--no-session-persistence", "--output-format", "json",
           "--json-schema", json.dumps(SCHEMA)]
    t0 = time.time()
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                             cwd=Path(__file__).parent)
    except subprocess.TimeoutExpired:
        return None, f"no answer within {TIMEOUT_S} seconds"
    log(f"  agent answered in {time.time() - t0:.0f} s")
    if run.returncode != 0:
        return None, f"claude stopped with an error: {(run.stderr or run.stdout).strip()[:200]}"
    try:
        outer = json.loads(run.stdout)
        data = outer.get("structured_output")
        if data is None:
            text = outer.get("result", "").strip().removeprefix("```json").removesuffix("```")
            data = json.loads(text)
        return data, None
    except (json.JSONDecodeError, AttributeError) as e:
        return None, f"the answer was not in the expected format ({e})"


# ---------------------------------------------------------------- 3. check and combine
def final_counts(result, decisions):
    counts = {g: 0 for g in result["grade_counts"]}
    for o in result["objects"]:
        action = decisions.get(o["id"], {}).get("action", "count_as_one")
        counts[o["grade"]] += {"count_as_two": 2, "not_a_plant": 0}.get(action, 1)
    return counts


def plain_answer(counts, n_flagged):
    parts = [f"{c} {g}" for g, c in counts.items() if c]
    total = sum(counts.values())
    text = f"There are {total} plants: " + (", ".join(parts) if parts else "none") + "."
    if n_flagged:
        text += (" 1 object needs a person to check." if n_flagged == 1
                 else f" {n_flagged} objects need a person to check.")
    return text


def fingerprint(result, expected):
    """A short code for 'these exact measurements and this expected count'."""
    keys = ("id", "height_cm", "spread_cm", "max_width_cm", "edge_distance_cm",
            "split_from_touching", "grade")
    data = {"v": 1, "expected": expected,
            "objects": [{k: o[k] for k in keys} for o in result["objects"]]}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def review(result, expected=None, image_path="output/inventory.png", use_agent=True, log=print,
           cache_dir=None):
    """cache_dir: if given, the agent's answer is saved there. The same measurements later
    reuse the saved answer, so that repeated runs give exactly the same result."""
    doubts, count_doubt = find_doubts(result, expected)
    log(f"Doubtful cases found by the rules: {len(doubts)}"
        + (f"; {count_doubt}" if count_doubt else ""))
    ids = {o["id"] for o in result["objects"]}

    data, problem, cached = None, "the agent was switched off", False
    cache_file = Path(cache_dir) / f"{fingerprint(result, expected)}.json" if cache_dir else None
    if use_agent and cache_file and cache_file.exists():
        data, cached = json.loads(cache_file.read_text()), True
        log("  same measurements as an earlier run: reusing the agent's saved answer")
    elif use_agent:
        log("Asking the agent (Claude) to judge them ...")
        data, problem = ask_claude(build_prompt(result, doubts, count_doubt, expected,
                                                str(Path(image_path).resolve())), log)
        if data is not None and cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, indent=1))

    if data is not None:
        decisions = {d["id"]: d for d in data["decisions"]
                     if d["id"] in ids and d["action"] in ACTIONS}
        counts = final_counts(result, decisions)
        agent_counts = {g["grade"]: g["count"] for g in data["grade_counts"]}
        n_flag = sum(d["action"] == "flag_for_person" for d in decisions.values())
        if {g: agent_counts.get(g, 0) for g in counts} == counts:
            answer, note = data["answer"], None
        else:
            answer = plain_answer(counts, n_flag)
            note = "The agent's counts did not match its own decisions; the program's count is shown."
        return {"agent_used": True, "agent_answer_reused": cached,
                "decisions": list(decisions.values()),
                "count_note": data["count_note"], "grade_counts": counts,
                "answer": answer, "check_note": note, "rule_doubts": doubts}

    # Fallback: no agent. Every doubtful case goes to a person; nothing is guessed.
    log(f"  agent not available: {problem}. Doubtful cases are flagged for a person.")
    decisions = [{"id": d["id"], "action": "flag_for_person",
                  "reason": "Automatic check: " + "; ".join(d["rule_reasons"]) + "."}
                 for d in doubts]
    counts = final_counts(result, {})
    return {"agent_used": False, "agent_problem": problem, "decisions": decisions,
            "count_note": count_doubt or "No count problem was found by the automatic check.",
            "grade_counts": counts, "answer": plain_answer(counts, len(decisions)),
            "check_note": None, "rule_doubts": doubts}


def main():
    ap = argparse.ArgumentParser(description="Agent check of output/inventory.json")
    ap.add_argument("--out", default="output", help="results folder. Default: output")
    ap.add_argument("--expected", type=int, help="how many plants you expect (optional)")
    ap.add_argument("--no-agent", action="store_true", help="do not call Claude (rules only)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "agent.json").unlink(missing_ok=True)
    src = out / "inventory.json"
    if not src.exists():
        raise SystemExit("No output/inventory.json. Run plant_inventory.py first.")
    result = json.loads(src.read_text())
    rev = review(result, args.expected, str(out / "inventory.png"), not args.no_agent)
    (out / "agent.json").write_text(json.dumps(rev, indent=1))

    print("\nDecisions:")
    for d in rev["decisions"]:
        print(f"  #{d['id']}: {d['action'].replace('_', ' ')}. {d['reason']}")
    if not rev["decisions"]:
        print("  none needed")
    print(f"\n{rev['count_note']}")
    print(f"\n{rev['answer']}")
    if rev["check_note"]:
        print(f"({rev['check_note']})")
    print(f"\nWrote {out}/agent.json")


if __name__ == "__main__":
    main()
