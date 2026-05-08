import json

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

# Cell 5: Run 4 training loop - skip to just runs 9-10
cell5 = nb["cells"][5]
old = "".join(cell5["source"])
if "NUM_RUNS = 10" in old:
    new = old.replace(
        "for run_num in range(1, NUM_RUNS + 1):",
        "# Runs 1-8 already done, completing 9-10\nfor run_num in range(9, NUM_RUNS + 1):",
    )
    cell5["source"] = [new]

# Cell 14: Run 5 training loop - add auto-save progress
cell14 = nb["cells"][14]
old14 = "".join(cell14["source"])
save_hook = """
    # auto-save progress
    import json as _js
    with open("run5_progress.json", "w") as _f:
        _js.dump({"best_acc": best5_acc, "best_run": best5_run, "runs": all5_metrics, "current_run": run_num}, _f)
"""
if "auto-save" not in old14:
    cell14["source"] = cell14["source"] + [save_hook]

with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done. Cells:", len(nb["cells"]))
print("All outputs cleared, auto-save added to Run 5")
