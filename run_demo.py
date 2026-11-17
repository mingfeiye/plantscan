"""
run_demo.py
Runs everything on one scan file and opens the report page:
  measure (plant_inventory) -> agent check (agent_review) -> report page (make_report) -> open.

Use the short command demo.sh (run in the project folder):
    ./demo.sh "scans/my scan.ply" --expected 9
    ./demo.sh backup --expected 9          # the known-good backup scan in demo/
If the scan is bad, it opens a page that says "Scan problem: no results" and shows no numbers.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import plant_inventory as pi          # noqa: E402
import agent_review as ar             # noqa: E402
import make_report as mr              # noqa: E402

BACKUP = HERE / "demo" / "backup_scan.ply"


def open_page(path):
    subprocess.run(["open", str(path)], check=False)        # macOS: open in the default browser


def main():
    ap = argparse.ArgumentParser(description="Scan file -> plant inventory report")
    ap.add_argument("scan", help='scan file, or the word "backup" for the backup scan')
    ap.add_argument("--expected", type=int, help="how many plants you put down (optional)")
    ap.add_argument("--no-agent", action="store_true", help="rules only, do not call Claude")
    ap.add_argument("--no-open", action="store_true", help="do not open the page (for tests)")
    args = ap.parse_args()

    t0 = time.time()
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    for name in ("inventory.csv", "inventory.json", "inventory.png", "agent.json", "report.html"):
        (out / name).unlink(missing_ok=True)                   # never show an old result
    scan = BACKUP if args.scan == "backup" else Path(args.scan).expanduser()
    page = out / "report.html"

    print(f"=== 1/3 Measuring {scan.name}")
    try:
        result = pi.analyse(str(scan), str(HERE / "config" / "grades.toml"))
    except Exception as e:
        msg = str(e) if isinstance(e, pi.ScanError) else \
            f"The scan could not be processed ({type(e).__name__}: {e})."
        print(f"\nSCAN PROBLEM: {msg}\nNo numbers are shown.")
        page.write_text(mr.make_problem_report(msg, str(scan)))
        if not args.no_open:
            open_page(page)
        sys.exit(2)
    pi.write_outputs(result, out)
    print(f"  {len(result['objects'])} objects measured")

    print("=== 2/3 Agent check")
    clean = json.loads((out / "inventory.json").read_text())
    rev = ar.review(clean, args.expected, str(out / "inventory.png"), not args.no_agent,
                    cache_dir=HERE / "cache" / "agent")
    (out / "agent.json").write_text(json.dumps(rev, indent=1))

    print("=== 3/3 Report")
    page.write_text(mr.make_report(clean, rev, out / "inventory.png"))
    print(f"\n{rev['answer']}")
    print(f"\nDone in {time.time() - t0:.1f} s. Report: {page}")
    if not args.no_open:
        open_page(page)


if __name__ == "__main__":
    main()
