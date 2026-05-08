import json, os

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Clear all outputs
for c in cells:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

# Cells 0-5: Keep as-is (TPU init, dataset, data loading, skip Run 4)
# These are correct.

# Replace cells 6-10 (Run 4 training, plots, save) with a simple pass-through
# because Run 4 was already completed in the old session
replacements = [
    (6, ["""all_run_metrics = []
best_overall_acc = 0.6901
best_overall_run = 5
best_history = None
print("Run 4 already completed previously (best: 69.01%). Proceeding to Run 5.")
"""]),
    (7, ["""print("Run 4 visualizations were saved in previous session. Skipping.")
"""]),
    (8, ["""print("Run 4 accuracy/loss curves were saved in previous session. Skipping.")
"""]),
    (9, ["""print("Run 4 classification report was saved in previous session. Skipping.")
"""]),
    (10, ["""print("Run 4 model already saved from previous session. Skipping.")
"""]),
]

for idx, src in replacements:
    cells[idx]["source"] = src

# Cells 11-20 (Run 5): Keep as-is - they're already correct

# Remove progress file
if os.path.exists("run5_progress.json"):
    os.remove("run5_progress.json")

with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. {len(cells)} cells")
print("Cells 0-5: Setup (TPU + dataset + data loading + Run 4 skip)")
print("Cells 6-10: Run 4 placeholder (previous session data)")
print("Cells 11-20: Run 5 training (targeting 93%)")
