"""`report <phase>` -> <phase>/REPORT.md (markdown tables by regime and by family: jobs, pass rate, value spread of the passers,
reasons histogram, compute used) and `ledger` -> results/md_campaign/LEDGER.md (append-only rows in the construct/LEDGER.md style).
Regime labels are descriptive; nothing here ranks or filters a technique."""
from __future__ import annotations
from typing import Dict, List
import collections, datetime, json, os, statistics
from . import ROOT, phase_dir, PHASES
from . import queue as Q


def _q(xs: List[float]) -> str:
    if not xs: return "-"
    xs = sorted(xs); return f"{xs[0]:.3g} / {statistics.median(xs):.3g} / {xs[-1]:.3g}"


def _group_table(results: List[dict], jobs: List[Q.Job], key: str, title: str) -> List[str]:
    by_r = collections.defaultdict(list); by_j = collections.defaultdict(list)
    for r in results: by_r[r.get(key) or "?"].append(r)
    for j in jobs: by_j[getattr(j, key) or "?"].append(j)
    L = [f"## By {title}", "", f"| {title} | jobs | done | failed | skipped | passed | pass rate | value min / median / max (passers) | top reasons | evals | CPU h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(set(by_r) | set(by_j)):
        rs = by_r.get(g, []); js = by_j.get(g, []); st = collections.Counter(j.state for j in js); ok = [r for r in rs if not r.get("error")]; ps = [r for r in ok if r.get("pass")]
        reasons = collections.Counter(x for r in ok if not r.get("pass") for x in r.get("reasons", [])); top = ", ".join(f"{k} x{v}" for k, v in reasons.most_common(4)) or "-"
        vals = [float(r["outcome"]["value"]) for r in ps if (r.get("outcome") or {}).get("value") is not None]
        L.append(f"| {g} | {len(js)} | {st.get('done', 0)} | {st.get('failed', 0)} | {st.get('skipped', 0)} | {len(ps)} | {(len(ps)/len(ok)*100 if ok else 0):.0f}% | {_q(vals)} | {top} | {sum(int(r.get('budget', 0)) for r in rs)} | {sum(float(r.get('secs', 0)) for r in rs)/3600:.2f} |")
    return L + [""]


def build_report(phase: str, smoke: bool = False) -> str:
    pdir = phase_dir(phase, smoke); results = Q.results(pdir); jobs = Q.load(pdir); st = collections.Counter(j.state for j in jobs)
    ok = [r for r in results if not r.get("error")]; ps = [r for r in ok if r.get("pass")]
    L = [f"# Campaign report: {phase}{' (SMOKE)' if smoke else ''}", "", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from `{os.path.relpath(pdir, ROOT)}` ({len(results)} result files; queue: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) + ").", "",
         "PASS = valid at both timesteps (1 ms and 0.5 ms replay, violation 0) AND the program's `achieved` flag. Regime labels describe what the problem asks of the support; they never filter. "
         "A program passes the phase when ANY of its jobs passed (that is what the next phase's generator reads). Values are each program's own scalar (m/s for projection/impact release speed, m for displacement, error for targeting: not comparable across programs).", "",
         f"**{len(ps)} / {len(ok)} executed jobs passed** ({len({r['program_id'] for r in ps})} programs); {st.get('failed', 0)} failed, {st.get('skipped', 0)} skipped; compute {sum(float(r.get('secs', 0)) for r in results)/3600:.2f} CPU h over {sum(int(r.get('budget', 0)) for r in results)} evaluations.", ""]
    L += _group_table(results, jobs, "regime", "regime"); L += _group_table(results, jobs, "family", "family")
    L += ["## Reasons histogram (executed, non-passing jobs)", "", "| reason | count |", "|---|---|"]
    for k, v in collections.Counter(x for r in ok if not r.get("pass") for x in r.get("reasons", [])).most_common(): L.append(f"| {k} | {v} |")
    L += ["", "## Per program", "", "| program | regime | family | robot | representation | jobs | passed | best value | valid both dt | achieved | reasons (best job) | arm | s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    byp = collections.defaultdict(list)
    for r in results: byp[(r["program_id"], r.get("arm") or "")].append(r)
    for (pid, arm) in sorted(byp):
        rs = byp[(pid, arm)]; best = max(rs, key=lambda r: ((r.get("pass") or False), float((r.get("outcome") or {}).get("value") or float("-inf")) if (r.get("outcome") or {}).get("value") is not None else float("-inf")))
        o = best.get("outcome") or {}; v = o.get("value")
        L.append(f"| {pid} | {best.get('regime')} | {best.get('family')} | {best.get('robot')} | {best.get('representation')} | {len(rs)} | {sum(1 for r in rs if r.get('pass'))} | {v if v is None else f'{float(v):.3f}'} | {best.get('valid_both_dt')} | {o.get('achieved')} | {', '.join(best.get('reasons', [])) or '-'} | {arm or '-'} | {sum(float(r.get('secs', 0)) for r in rs):.0f} |")
    sk = [j for j in jobs if j.state in ("skipped", "failed")]
    if sk:
        L += ["", "## Skipped / failed jobs", "", "| job | state | reason |", "|---|---|---|"] + [f"| {j.id} | {j.state} | {j.reason[:200]} |" for j in sk]
    return "\n".join(L) + "\n"


def write_report(phase: str, smoke: bool = False) -> str:
    pdir = phase_dir(phase, smoke); p = os.path.join(pdir, "REPORT.md"); open(p, "w").write(build_report(phase, smoke)); return p


# ---------------------------------------------------------------- ledger
LEDGER = os.path.join(ROOT, "LEDGER.md")
HEAD = ["# movement_discovery campaign ledger (auto-generated by movement_discovery.campaign ledger)", "",
        "One row per executed campaign job (results/md_campaign/<phase>/<job>.json). value = the program's own scalar; valid = both timesteps pass every gate; pass = valid AND achieved. Regime/ownership/order are descriptive. Smoke rows (budget override) are excluded.", "",
        "| phase | program | regime | family | scenario | robot | rep | seed | evals | value | valid | achieved | pass | reasons | peaks (rad/s) | order | ownership | beyond human | arm | s | job |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def _ledger_ids() -> set:
    if not os.path.exists(LEDGER): return set()
    return {l.rsplit("|", 2)[-2].strip() for l in open(LEDGER) if l.startswith("| ") and not l.startswith("| phase")}


def append_ledger(phases: List[str] = PHASES) -> int:
    have = _ledger_ids(); rows = []
    for ph in phases:
        pdir = os.path.join(ROOT, ph)
        if not os.path.isdir(pdir): continue
        for r in Q.results(pdir):
            if r["job"] in have or r.get("budget_override"): continue
            o = r.get("outcome") or {}; m = r.get("mechanics") or {}; v = o.get("value")
            own = {k: round(float(x), 2) for k, x in (m.get("ownership_group") or {}).items() if isinstance(x, (int, float)) and abs(x) >= 0.3} if isinstance(m, dict) else {}
            rows.append(f"| {ph} | {r['program_id']} | {r.get('regime')} | {r.get('family')} | {r['scenario_id']} | {r['robot']} | {Q.rep_short(r['representation'])} | {r['seed']} | {r.get('budget')} | {'-' if v is None else f'{float(v):.3f}'} | {'yes' if r.get('valid_both_dt') else 'NO'} | {o.get('achieved', '-')} | {'yes' if r.get('pass') else 'NO'} | {','.join(r.get('reasons', [])) or '-'} | {m.get('peak_omega') if isinstance(m, dict) else '-'} | {('p->d' if m.get('strict_proximal_to_distal') else 'mixed') if isinstance(m, dict) and m else '-'} | {own or '-'} | {','.join(m.get('beyond_human') or []) if isinstance(m, dict) else '-'} | {r.get('arm') or '-'} | {float(r.get('secs', 0)):.0f} | {r['job']} |")
    new = not os.path.exists(LEDGER)
    if not rows and new: return 0
    with open(LEDGER, "a") as f:
        if new: f.write("\n".join(HEAD) + "\n")
        for row in rows: f.write(row + "\n")
    return len(rows)
