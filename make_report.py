"""
make_report.py
Makes ONE report page (output/report.html) from the results of plant_inventory.py and
agent_review.py. The page is a single file: the image is inside it, so it opens without
internet and can be copied anywhere. Large text, so it can be read from 2 meters away.

Use (run in the project folder, after the two other programs):
    venv/bin/python make_report.py
If the scan was bad, make_problem_report() makes a page that says so, with no numbers.
"""
import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path

CSS = """
:root { --ink:#111; --muted:#555; --line:#ccc; --bg:#fff; --soft:#f3f3f0;
        --ok:#1b6e2e; --okbg:#e3f3e6; --warn:#8a4b00; --warnbg:#fff0d6;
        --bad:#a1161b; --badbg:#fde4e4; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:22px/1.4 -apple-system, "Helvetica Neue", Arial, sans-serif; }
main { max-width:1500px; margin:0 auto; padding:28px 32px 48px; }
h1 { font-size:46px; margin:0 0 4px; }
h2 { font-size:30px; margin:36px 0 14px; border-bottom:3px solid var(--ink); padding-bottom:6px; }
.meta { color:var(--muted); font-size:20px; }
.answer { font-size:30px; margin:22px 0 8px; padding:18px 22px; background:var(--soft);
          border-left:10px solid var(--ink); }
.tiles { display:flex; flex-wrap:wrap; gap:16px; margin-top:18px; }
.tile { flex:1 1 200px; border:3px solid var(--ink); padding:12px 16px; text-align:center; }
.tile .n { font-size:72px; font-weight:800; line-height:1.05; }
.tile .g { font-size:24px; font-weight:600; }
.tile.total { background:var(--ink); color:var(--bg); }
.tile.flag { border-color:var(--warn); color:var(--warn); background:var(--warnbg); }
.cols { display:flex; flex-wrap:wrap; gap:32px; align-items:flex-start; }
.cols > section:first-child { flex:1 1 480px; min-width:0; }
.cols > section:last-child { flex:1 1 640px; min-width:0; overflow-x:auto; }
img { width:100%; height:auto; border:2px solid var(--line); }
table { width:100%; border-collapse:collapse; font-size:28px; }
th, td { padding:8px 12px; border-bottom:2px solid var(--line); text-align:right; white-space:nowrap; }
th { font-size:22px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
td.l, th.l { text-align:left; }
td.id { font-weight:800; }
tr.flagged td { background:var(--warnbg); }
.badge { display:inline-block; padding:2px 12px; border-radius:6px; font-size:20px;
         font-weight:700; white-space:nowrap; }
.b-ok { background:var(--okbg); color:var(--ok); }
.b-flag { background:var(--warnbg); color:var(--warn); border:2px solid var(--warn); }
.b-two { background:#e4ecfb; color:#173f8a; }
.b-no { background:#eee; color:#444; }
ul.dec { list-style:none; padding:0; margin:0; }
ul.dec li { padding:12px 0; border-bottom:2px solid var(--line); font-size:26px; }
ul.dec b { font-size:30px; margin-right:8px; }
.note { color:var(--muted); font-size:20px; margin-top:28px; }
.demo { color:var(--bad); font-weight:700; }
.problem { margin-top:40px; padding:36px; border:6px solid var(--bad); background:var(--badbg);
           color:var(--bad); }
.problem h2 { font-size:52px; border:none; margin:0 0 12px; color:var(--bad); }
.problem p { font-size:32px; margin:8px 0; color:var(--ink); }
"""

LABEL = {"count_as_one": ("1 plant", "b-ok"), "count_as_two": ("2 plants", "b-two"),
         "not_a_plant": ("Not a plant", "b-no"), "flag_for_person": ("Check", "b-flag")}


def page(title, body):
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")


def esc(x):
    return html.escape(str(x))


def make_report(inv, agent, image_path):
    objs = inv["objects"]
    decisions = {d["id"]: d for d in (agent or {}).get("decisions", [])}
    counts = (agent or {}).get("grade_counts") or inv["grade_counts"]
    total = sum(counts.values())
    n_flag = sum(d["action"] == "flag_for_person" for d in decisions.values())
    img64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    when = datetime.now().strftime("%Y-%m-%d %H:%M")

    tiles = f"<div class='tile total'><div class='n'>{total}</div><div class='g'>plants in total</div></div>"
    for g, c in counts.items():
        tiles += f"<div class='tile'><div class='n'>{c}</div><div class='g'>{esc(g)}</div></div>"
    if n_flag:
        tiles += (f"<div class='tile flag'><div class='n'>{n_flag}</div>"
                  f"<div class='g'>to check by a person</div></div>")

    rows = ""
    for o in objs:
        d = decisions.get(o["id"])
        lab, cls = LABEL[d["action"]] if d else ("OK", "b-ok")
        rows += (f"<tr class='{'flagged' if d and d['action'] == 'flag_for_person' else ''}'>"
                 f"<td class='id l'>{o['id']}</td><td>{o['height_cm']:.1f}</td>"
                 f"<td>{o['spread_cm']:.1f}</td><td class='l'>{esc(o['grade'])}</td>"
                 f"<td class='l'><span class='badge {cls}'>{lab}</span></td></tr>")

    if decisions:
        dec = "".join(f"<li><b>#{d['id']}</b><span class='badge {LABEL[d['action']][1]}'>"
                      f"{LABEL[d['action']][0]}</span> {esc(d['reason'])}</li>"
                      for d in sorted(decisions.values(), key=lambda d: d["id"]))
    else:
        dec = "<li>No doubtful cases. Every object was counted as one plant.</li>"

    if agent is None:
        who = "The agent step did not run."
    elif agent.get("agent_used"):
        who = "Decisions by the agent (Claude), checked by the program."
        if agent.get("agent_answer_reused"):
            who += " (Same measurements as an earlier run, so the agent's earlier answer is reused.)"
    else:
        who = (f"The agent was not available ({esc(agent.get('agent_problem', ''))}), so every "
               f"doubtful case is flagged for a person.")
    answer = esc((agent or {}).get("answer") or f"There are {total} plants.")
    count_note = esc((agent or {}).get("count_note", ""))
    check_note = (agent or {}).get("check_note")
    demo = ("<span class='demo'>Grade limits are DEMO values, not ANSI Z60.1 values.</span> "
            if inv.get("grades_are_demo_values") else "")

    body = f"""
<h1>Plant inventory</h1>
<div class='meta'>Scan: {esc(Path(inv['scan']).name)} &nbsp;·&nbsp; {when}</div>
<div class='answer'>{answer}</div>
<div class='tiles'>{tiles}</div>
<div class='cols'>
  <section><h2>Top view</h2>
    <img alt='Top view of the scan with each plant numbered' src='data:image/png;base64,{img64}'>
  </section>
  <section><h2>Measurements</h2>
    <table><thead><tr><th class='l'>#</th><th>Height (cm)</th><th>Spread (cm)</th>
      <th class='l'>Grade</th><th class='l'>Check</th></tr></thead><tbody>{rows}</tbody></table>
  </section>
</div>
<h2>Agent decisions</h2>
<p class='meta'>{who} {count_note}{(' ' + esc(check_note)) if check_note else ''}</p>
<ul class='dec'>{dec}</ul>
<p class='note'>{demo}Height is measured from the surface to the top of the plant; spread is the
average width seen from above (for a round plant, its diameter). Phone scan accuracy is about
1 cm. Trunk caliper is not measured.</p>
"""
    return page("Plant inventory", body)


def make_problem_report(message, scan=""):
    """A page for a bad scan: it says so clearly and shows NO numbers."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = f"""
<h1>Plant inventory</h1>
<div class='meta'>Scan: {esc(Path(scan).name) if scan else '-'} &nbsp;·&nbsp; {when}</div>
<div class='problem'><h2>Scan problem: no results</h2>
<p>{esc(message)}</p>
<p>No numbers are shown, because they would not be reliable. Please scan again, or run the
backup scan.</p></div>
"""
    return page("Plant inventory: scan problem", body)


def main():
    ap = argparse.ArgumentParser(description="Make output/report.html")
    ap.add_argument("--out", default="output", help="results folder. Default: output")
    args = ap.parse_args()
    out = Path(args.out)
    inv = json.loads((out / "inventory.json").read_text())
    agent_file = out / "agent.json"
    agent = json.loads(agent_file.read_text()) if agent_file.exists() else None
    (out / "report.html").write_text(make_report(inv, agent, out / "inventory.png"))
    print(f"Wrote {out}/report.html")


if __name__ == "__main__":
    main()
