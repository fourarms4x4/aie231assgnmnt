import json

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

# Cell 5: Run 4 training loop - skip entirely
cell5 = nb["cells"][5]
cell5["source"] = ['print("Run 4 already completed in previous session. Skipping to Run 5.")\n']

# Cell 14: Run 5 training loop - reduce to 5 runs x 30 epochs
cell14 = nb["cells"][14]
old14 = "".join(cell14["source"])

# Reduce settings
old14 = old14.replace("NUM_RUNS_5 = 10", "NUM_RUNS_5 = 5")
old14 = old14.replace("EPOCHS_5 = 100", "EPOCHS_5 = 30")

# Add auto-save hook
save_hook = """
    # auto-save progress
    import json as _js
    with open("run5_progress.json", "w") as _f:
        _js.dump({"best_acc": best5_acc, "best_run": best5_run, "runs": all5_metrics, "current_run": run_num}, _f)
"""
if "auto-save" not in old14:
    old14 += save_hook

cell14["source"] = [old14]

with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
if os.path.exists("run5_progress.json"):
    os.remove("run5_progress.json")

print("Done. Cells:", len(nb["cells"]))
print("Run 4: SKIPPED")
print("Run 5: 5 runs x 30 epochs (fast mode)")
