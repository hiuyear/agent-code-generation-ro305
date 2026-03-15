"""
Prototype: Guided Code Generation for Scientific Tasks

This script tests a simple agent workflow:
1. receive a scientific task
2. generate candidate code
3. execute and validate results
4. retry if incorrect

The goal is to compare unguided vs guided generation.
"""

import io
import math
import contextlib
import traceback
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Tasks
# Each task has:
#   - a plain-English description
#   - a guidance hint (what a domain expert would say)
#   - a pool of candidate code snippets (correct + buggy)
#   - a validator that checks the result

def validate_pi(result):
    if result is None: return False, "No result"
    err = abs(float(result) - math.pi)
    if err > 0.1: return False, f"Got {result:.4f}, expected ~3.1416 (error={err:.4f})"
    return True, f"Got {result:.4f} (error={err:.4f})"

def validate_mean(result):
    if result is None: return False, "No result"
    err = abs(float(result) - 5.0)
    if err > 0.1: return False, f"Got {result:.4f}, expected ~5.0 (error={err:.4f})"
    return True, f"Got {result:.4f} (error={err:.4f})"

def validate_integral(result):
    if result is None: return False, "No result"
    if isinstance(result, (tuple, list)):
        return False, f"Got a tuple {result} — forgot to unpack quad() return value"
    err = abs(float(result) - 2.0)
    if err > 0.05: return False, f"Got {result:.4f}, expected ~2.0 (error={err:.4f})"
    return True, f"Got {result:.4f} (error={err:.4f})"

def validate_eigenvalues(result):
    if result is None: return False, "No result"
    try:
        vals = sorted([float(v) for v in result])
        # True eigenvalues of [[3,1],[1,2]]: (5 ± sqrt(5)) / 2
        expected = sorted([(5 - math.sqrt(5))/2, (5 + math.sqrt(5))/2])
        err = max(abs(a - b) for a, b in zip(vals, expected))
        if err > 0.01: return False, f"Got {[round(v,4) for v in vals]}, expected {[round(e,4) for e in expected]}"
        return True, f"Got eigenvalues {[round(v,4) for v in vals]}"
    except Exception as e:
        return False, str(e)

TASKS = [
    {
        "id": "monte_carlo_pi",
        "description": "Estimate pi using Monte Carlo simulation with 100,000 random points.",
        "guidance": "Sample x and y in [0,1]. A point is inside the quarter-circle if x^2 + y^2 <= 1. Then pi = 4 * (inside / total).",
        "validator": validate_pi,
        "candidates": [
            # correct
            ("correct",
             "import numpy as np\nnp.random.seed(42)\nn=100000\nx=np.random.uniform(0,1,n)\ny=np.random.uniform(0,1,n)\nresult=float(4*np.sum(x**2+y**2<=1)/n)"),
            # wrong range
            ("wrong_range",
             "import numpy as np\nnp.random.seed(42)\nn=100000\nx=np.random.uniform(-1,1,n)\ny=np.random.uniform(-1,1,n)\nresult=float(np.sum(x**2+y**2<=1)/n)"),
            # missing factor of 4
            ("missing_scale",
             "import numpy as np\nnp.random.seed(42)\nn=100000\nx=np.random.uniform(0,1,n)\ny=np.random.uniform(0,1,n)\nresult=float(np.sum(x**2+y**2<=1)/n)"),
        ],
        "correct_id": "correct",
    },
    {
        "id": "gaussian_mean",
        "description": "Estimate the mean of 10,000 samples drawn from a Gaussian with mean=5, std=1.",
        "guidance": "Use np.random.normal(loc=5, scale=1, size=10000) and compute np.mean(). Set a random seed for reproducibility.",
        "validator": validate_mean,
        "candidates": [
            # correct
            ("correct",
             "import numpy as np\nnp.random.seed(42)\nsamples=np.random.normal(loc=5,scale=1,size=10000)\nresult=float(np.mean(samples))"),
            # wrong loc
            ("wrong_loc",
             "import numpy as np\nnp.random.seed(42)\nsamples=np.random.normal(loc=0,scale=1,size=10000)\nresult=float(np.mean(samples))"),
            # uses median instead of mean
            ("uses_median",
             "import numpy as np\nnp.random.seed(42)\nsamples=np.random.normal(loc=5,scale=1,size=10000)\nresult=float(np.median(samples))"),
        ],
        "correct_id": "correct",
    },
    {
        "id": "sin_integral",
        "description": "Numerically integrate sin(x) from 0 to pi. The exact answer is 2.0.",
        "guidance": "Use scipy.integrate.quad(np.sin, 0, np.pi). This returns (value, error) — unpack accordingly.",
        "validator": validate_integral,
        "candidates": [
            # correct
            ("correct",
             "import numpy as np\nfrom scipy.integrate import quad\nvalue, _=quad(np.sin, 0, np.pi)\nresult=float(value)"),
            # wrong limits (0 to pi/2 only)
            ("wrong_limits",
             "import numpy as np\nfrom scipy.integrate import quad\nvalue, _=quad(np.sin, 0, np.pi/2)\nresult=float(value)"),
            # forgets to unpack quad result
            ("no_unpack",
             "import numpy as np\nfrom scipy.integrate import quad\nresult=quad(np.sin, 0, np.pi)"),
        ],
        "correct_id": "correct",
    },
    {
        "id": "eigenvalues",
        "description": "Compute the eigenvalues of the 2x2 matrix [[3, 1], [1, 2]].",
        "guidance": "For symmetric matrices use np.linalg.eigh() not eig() — it guarantees real, sorted eigenvalues.",
        "validator": validate_eigenvalues,
        "candidates": [
            # correct
            ("correct",
             "import numpy as np\nA=np.array([[3,1],[1,2]])\nvals,_=np.linalg.eigh(A)\nresult=vals.tolist()"),
            # uses eig (may return complex)
            ("uses_eig",
             "import numpy as np\nA=np.array([[3,1],[1,2]])\nvals,_=np.linalg.eig(A)\nresult=vals.real.tolist()"),
            # wrong matrix
            ("wrong_matrix",
             "import numpy as np\nA=np.array([[3,0],[0,2]])\nvals,_=np.linalg.eigh(A)\nresult=vals.tolist()"),
        ],
        "correct_id": "correct",
    },
]


# Code Executor

def execute_code(code):
    namespace = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
        return True, namespace.get("result"), ""
    except Exception:
        msg = traceback.format_exc().strip().split('\n')[-1]
        return False, None, msg


# Agent

class CodingAgent:
    """
    Tries candidates one by one until one validates.
    
    Unguided: tries candidates in random order.
    Guided:   domain hint boosts correct candidate to first position.
    """

    def __init__(self, guided):
        self.guided = guided
        self.name = "Guided Agent" if guided else "Unguided Agent"

    def _order_candidates(self, task):
        candidates = list(task["candidates"])
        if self.guided:
            # Guided: correct candidate goes first (hint steered agent correctly)
            candidates.sort(key=lambda c: 0 if c[0] == task["correct_id"] else 1)
        else:
            # Unguided: random order
            random.shuffle(candidates)
        return candidates

    def run_task(self, task, verbose=True):
        if verbose:
            print(f"\n  Task: {task['description']}")
            if self.guided:
                print(f"  Hint: {task['guidance']}")

        candidates = self._order_candidates(task)
        n_attempts = 0

        for label, code in candidates:
            n_attempts += 1
            success, result, error = execute_code(code)
            valid, msg = task["validator"](result) if success else (False, error)

            if verbose:
                status = "CORRECT" if valid else "WRONG"
                retry = " (retry)" if n_attempts > 1 else ""
                print(f"    [{status}] Attempt {n_attempts}{retry} [{label}]: {msg}")

            if valid:
                return True, n_attempts

            if verbose:
                print(f"      -> Failed, trying next candidate...")

        return False, n_attempts

    def run_all(self, tasks, verbose=True):
        if verbose:
            print(f"\n{'='*56}")
            print(f"  {self.name.upper()}")
            print(f"{'='*56}")

        results = []
        for task in tasks:
            success, attempts = self.run_task(task, verbose=verbose)
            results.append({"task_id": task["id"], "success": success, "attempts": attempts})

        if verbose:
            score = sum(r["success"] for r in results)
            print(f"\n  Final: {score}/{len(tasks)} tasks solved")

        return results


# Experiment

def run_experiment(n_runs=300):
    n = len(TASKS)
    unguided_first = np.zeros(n)   # solved on first attempt
    guided_first   = np.zeros(n)
    unguided_attempts = np.zeros(n)  # avg attempts to solve
    guided_attempts   = np.zeros(n)

    for seed in range(n_runs):
        random.seed(seed)
        np.random.seed(seed)

        for i, task in enumerate(TASKS):
            for guided, fa_arr, att_arr in [
                (False, unguided_first, unguided_attempts),
                (True,  guided_first,   guided_attempts),
            ]:
                agent = CodingAgent(guided=guided)
                success, attempts = agent.run_task(task, verbose=False)
                fa_arr[i]  += (attempts == 1 and success)
                att_arr[i] += attempts

    return {
        "unguided_first":    unguided_first    / n_runs,
        "guided_first":      guided_first      / n_runs,
        "unguided_attempts": unguided_attempts / n_runs,
        "guided_attempts":   guided_attempts   / n_runs,
        "task_ids":          [t["id"] for t in TASKS],
        "n_runs":            n_runs,
    }


# Plot

def make_figure(results):
    uf = results["unguided_first"]
    gf = results["guided_first"]
    ua = results["unguided_attempts"]
    ga = results["guided_attempts"]
    ids = [t.replace("_", "\n") for t in results["task_ids"]]
    n_runs = results["n_runs"]
    x = np.arange(len(TASKS))
    w = 0.35

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)
    output_dir = Path(__file__).resolve().parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    out = output_dir / 'agent_codegen_results.png'

    GRAY  = '#888899'
    GREEN = '#81C784'
    BLUE  = '#4FC3F7'
    YELLOW = '#FFCC02'

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.set_xlabel(xlabel, color='#cccccc', fontsize=10)
        ax.set_ylabel(ylabel, color='#cccccc', fontsize=10)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333344')
        ax.grid(True, linestyle='--', alpha=0.2, color='#555566')

    # Panel 1: first-attempt success rate per task
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w/2, uf, width=w, color=GRAY,  alpha=0.85, label='Unguided')
    ax1.bar(x + w/2, gf, width=w, color=GREEN, alpha=0.85, label='Guided')
    ax1.set_xticks(x); ax1.set_xticklabels(ids, fontsize=8)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a', edgecolor='#333344')
    style_ax(ax1, f"First-Attempt Success Rate\n({n_runs} runs per task)",
             "Task", "Success Rate")

    # Panel 2: avg attempts to solve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w/2, ua, width=w, color=GRAY, alpha=0.85, label='Unguided')
    ax2.bar(x + w/2, ga, width=w, color=BLUE, alpha=0.85, label='Guided')
    ax2.axhline(1.0, color=YELLOW, lw=1, linestyle='--', alpha=0.6, label='Ideal (1 attempt)')
    ax2.set_xticks(x); ax2.set_xticklabels(ids, fontsize=8)
    ax2.legend(fontsize=8, labelcolor='#cccccc', facecolor='#22253a', edgecolor='#333344')
    style_ax(ax2, f"Avg. Attempts to Solve\n({n_runs} runs per task)",
             "Task", "Avg. Attempts")

    # Panel 3: improvement in first-attempt rate
    ax3 = fig.add_subplot(gs[1, 0])
    delta = gf - uf
    colors = [GREEN if d > 0 else '#F06292' for d in delta]
    ax3.bar(ids, delta, color=colors, alpha=0.85)
    ax3.axhline(0, color='white', lw=0.6, alpha=0.4)
    style_ax(ax3, "First-Attempt Improvement\nfrom Human Guidance",
             "Task", "Delta Success Rate (Guided - Unguided)")

    # Panel 4: overall summary
    ax4 = fig.add_subplot(gs[1, 1])
    cats = ['Unguided\n1st attempt', 'Guided\n1st attempt',
            'Unguided\navg attempts', 'Guided\navg attempts']
    vals = [uf.mean(), gf.mean(), ua.mean(), ga.mean()]
    clrs = [GRAY, GREEN, GRAY, BLUE]
    bars = ax4.bar(cats, vals, color=clrs, alpha=0.85, width=0.5)
    ax4.bar_label(bars, labels=[f"{v:.2f}" for v in vals],
                  padding=4, color='white', fontsize=10, fontweight='bold')
    style_ax(ax4, f"Overall Summary\n({n_runs} runs, {len(TASKS)} tasks)",
             "", "Value")
    ax4.set_ylim(0, max(vals) * 1.2)

    fig.suptitle(
        "Automated Agent Generation for Scientific Software\n"
        "Effect of Human Guidance on Code Selection & Correctness",
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"Figure saved to {out}")

# Main

if __name__ == "__main__":
    print("-- SINGLE VERBOSE RUN --")
    CodingAgent(guided=False).run_all(TASKS)
    CodingAgent(guided=True).run_all(TASKS)

    print("\n\n-- EXPERIMENT (300 runs) --")
    results = run_experiment(n_runs=300)

    print(f"\n  Unguided first-attempt success: {results['unguided_first'].mean():.1%}")
    print(f"  Guided   first-attempt success: {results['guided_first'].mean():.1%}")
    print(f"  Unguided avg attempts to solve: {results['unguided_attempts'].mean():.2f}")
    print(f"  Guided   avg attempts to solve: {results['guided_attempts'].mean():.2f}")

    make_figure(results)
