import json
nb = json.load(open("aie231assgnmnt3.ipynb", "r", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]) if c["source"] else ""
    ct = c["cell_type"]
    s = src[:80].replace("\n", "\\n")
    print(f"  [{i:2d}] {ct:8s} | {s}")
