import json, os

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Clear all cell outputs
for c in cells:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

# Cell 0: Replace nvidia-smi with TPU detection + fallback
cells[0]["source"] = [
    'import os, tensorflow as tf\n',
    'try:\n',
    '    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()\n',
    '    tf.config.experimental_connect_to_cluster(resolver)\n',
    '    tf.tpu.experimental.initialize_tpu_system(resolver)\n',
    '    strategy = tf.distribute.TPUStrategy(resolver)\n',
    '    print("TPU available!")\n',
    '    BATCH_SIZE = 128\n',
    'except (ValueError, tf.errors.NotFoundError):\n',
    '    print("No TPU. Using GPU/CPU.")\n',
    '    gpus = tf.config.list_physical_devices("GPU")\n',
    '    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()\n',
    '    BATCH_SIZE = 32\n',
    '    !nvidia-smi\n',
]

# Cell 4: Remove .cache() for TPU compat, dynamic BATCH_SIZE
cells[4]["source"] = [
    'import matplotlib.pyplot as plt\n',
    'import numpy as np\n',
    'from tensorflow.keras import layers, models, callbacks\n',
    'from sklearn.metrics import classification_report, confusion_matrix\n',
    'import seaborn as sns\n',
    'import datetime\n',
    '\n',
    'RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n',
    'RUN_DIR = f"./visualizations/run4_{RUN_TIMESTAMP}"\n',
    'os.makedirs(RUN_DIR, exist_ok=True)\n',
    '\n',
    'data_dir = "./egyptian-new-currency-2023/dataset"\n',
    'train_dir = os.path.join(data_dir, "train")\n',
    'IMG_HEIGHT = 224\n',
    'IMG_WIDTH = 224\n',
    '\n',
    'train_ds = tf.keras.utils.image_dataset_from_directory(\n',
    '    train_dir, validation_split=0.2, subset="training",\n',
    '    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)\n',
    'val_ds = tf.keras.utils.image_dataset_from_directory(\n',
    '    train_dir, validation_split=0.2, subset="validation",\n',
    '    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)\n',
    '\n',
    'class_names = train_ds.class_names\n',
    'NUM_CLASSES = len(class_names)\n',
    'print(f"Classes: {class_names}")\n',
    'print(f"Number of classes: {NUM_CLASSES}")\n',
    'AUTOTUNE = tf.data.AUTOTUNE\n',
    'train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)\n',
    'val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)\n',
]

# Cell 5: Skip Run 4
cells[5]["source"] = ['print("Run 4 already completed previously. Skipping to Run 5.")\n']

# Fix f-string issues in Run 4 cells (cells 6-10)
for i in [6, 7, 8, 9, 10]:
    c = cells[i]
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        for old_k, new_k in [
            ("m['run']", "m.get('run')"),
            ("m['best_val_acc']", "m.get('best_val_acc')"),
            ("m['best_epoch']", "m.get('best_epoch')"),
            ("m['final_val_acc']", "m.get('final_val_acc')"),
            ("m['final_train_acc']", "m.get('final_train_acc')"),
            ("m['epochs_run']", "m.get('epochs_run')"),
        ]:
            src = src.replace(old_k, new_k)
        c["source"] = [src]

# Cell 10: Markdown for Run 5 (index 10 is currently the Run 4 save cell)
# Don't change it - it's the Run 4 save code and should remain
# Instead, find the Run 5 markdown cell and update it
# Let me find the right index

# Find Run 5 markdown cell
run5_md_idx = None
for i in range(11, len(cells)):
    if cells[i]["cell_type"] == "markdown" and "RUN 5" in "".join(cells[i].get("source", [])):
        run5_md_idx = i
        break
if run5_md_idx:
    cells[run5_md_idx]["source"] = [
        "## RUN 5: IMPROVED ARCHITECTURE TARGETING 93%\n",
        "### Config:\n",
        "- 2 Conv layers per block (deeper): 32->64->128->256\n",
        "- TPU-optimized: 5 sub-runs x 30 epochs, batch_size=128 on TPU\n",
        "- SGD+momentum + CosineDecay LR (0.01 -> 1e-5)\n",
        "- Label smoothing (0.1)\n",
        "- Ensemble top 3 runs\n",
    ]

# Find and update Run 5 code cells with strategy.scope() and reduced settings
for i in range(11, len(cells)):
    c = cells[i]
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])

    # Reduce epochs and runs
    src = src.replace("NUM_RUNS_5 = 10", "NUM_RUNS_5 = 5")
    src = src.replace("EPOCHS_5 = 100", "EPOCHS_5 = 30")

    # Wrap model creation in strategy.scope() - approach 1: in the training loop
    src = src.replace(
        "m = create_model_v5(NUM_CLASSES)\n    m.compile(",
        "with strategy.scope():\n            m = create_model_v5(NUM_CLASSES)\n        m.compile(",
    )

    # Fix f-string issues (same as Run 4)
    for old_k, new_k in [
        ("m['run']", "m.get('run')"),
        ("m2['run']", "m2.get('run')"),
        ("m['best']", "m.get('best')"),
        ("m['epoch']", "m.get('epoch')"),
        ("m['final_val']", "m.get('final_val')"),
        ("m['final_train']", "m.get('final_train')"),
        ("m['epochs']", "m.get('epochs')"),
    ]:
        src = src.replace(old_k, new_k)

    c["source"] = [src]

# Remove old progress file
if os.path.exists("run5_progress.json"):
    os.remove("run5_progress.json")

with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Cells: {len(cells)}")
print(" - TPU detection + GPU fallback")
print(" - .cache() removed, TPU-compatible datasets")
print(" - strategy.scope() around model creation")
print(" - Run 4 skipped, Run 5: 5 sub-runs x 30 epochs")
print(" - All f-string dict access fixed")
