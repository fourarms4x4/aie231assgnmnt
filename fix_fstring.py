import json

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

fix_count = 0
for c in nb["cells"]:
    src = "".join(c["source"])
    # The bug: {[x[0]:.4f for x in top3_models]} - colon confuses f-string parser
    # Fix: just remove the :.4f formatting
    old = "{[x[0]:.4f for x in top3_models]}"
    new = "{[round(x[0], 4) for x in top3_models]}"
    if old in src:
        src = src.replace(old, new)
        fix_count += 1
    c["source"] = [src]

with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed {fix_count} occurrences")
