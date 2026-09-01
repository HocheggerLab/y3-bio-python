"""Which of these five neurons cares about the go cue?

Lab 5, session 3. Every neuron in `data/` is one cell recorded while a mouse
did a whisker task. This program loads all of them, measures how strongly each
one responds to the go cue, and ranks them.

Run it with:

    uv run analyse_neurons.py

REFERENCE SOLUTION — the students' copy is analyse_neurons.py, with the loop removed.
"""

from pathlib import Path

import numpy as np

FRAME_RATE = 10.8
BEFORE, AFTER = 10, 30          # frames kept either side of a cue

# Column order in the CSVs: frame, dff, cue, go, nogo, reward, punish, lick
COLUMNS = {"dff": 1, "cue": 2, "go": 3, "nogo": 4, "reward": 5}


# ---------------------------------------------------------------------------
# GIVEN — you wrote all of this in the notebooks.
# ---------------------------------------------------------------------------

def load_neuron(path):
    """Read one recording. Returns the trace and a dict of event markers."""
    columns = tuple(COLUMNS.values())
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, unpack=True)
    named = dict(zip(COLUMNS, values))
    return named.pop("dff"), named


def onsets_of(marker):
    """The frames where a marker switches 0 -> 1."""
    return np.where(np.diff(marker) == 1)[0] + 1


def trial_table(trace, marker):
    """One row per cue, one column per frame. Skips cues too close to either end."""
    starts = onsets_of(marker)
    windows = [trace[s - BEFORE:s + AFTER] for s in starts
               if s >= BEFORE and s + AFTER <= len(trace)]
    return np.stack(windows)


def go_response(trace, marker):
    """Peak of the baseline-corrected average trial, and when it happens."""
    trials = trial_table(trace, marker)
    corrected = trials - trials[:, :BEFORE].mean(axis=1)[:, np.newaxis]
    average = corrected.mean(axis=0)
    after = average[BEFORE:]
    return len(trials), after.max(), after.argmax() / FRAME_RATE


# ---------------------------------------------------------------------------
# YOUR JOB — find the data, run the analysis over all of it, rank the answers.
# ---------------------------------------------------------------------------

# TODO 1 — point this at the data folder you downloaded.
DATA_DIR = Path(__file__).parent / "data"

# TODO 2 — stop with a clear message if it isn't there.
if not DATA_DIR.exists():
    raise SystemExit(f"I can't find {DATA_DIR} — check the path in TODO 1.")

# TODO 3 — loop over every .csv, measure each neuron, collect the results.
results = []
for path in sorted(DATA_DIR.glob("*.csv")):
    trace, events = load_neuron(path)
    n_trials, peak, peak_at = go_response(trace, events["go"])
    results.append((path.stem, n_trials, peak, peak_at))

# TODO 4 — rank them, strongest responder first.
results.sort(key=lambda row: row[2], reverse=True)

print(f"{len(results)} neurons, ranked by their response to the go cue\n")
print(f"{'neuron':<28}{'trials':>7}{'peak':>8}{'at':>7}")
for name, n_trials, peak, peak_at in results:
    print(f"{name:<28}{n_trials:>7}{peak:>8.3f}{peak_at:>6.1f}s")
