import json, os, shutil

with open("aie231assgnmnt3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Clear all outputs
for c in cells:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

# Cell 0: TPU detection + GPU fallback
cells[0]["source"] = [
    'import os, tensorflow as tf\n',
    'try:\n',
    '    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()\n',
    '    tf.config.experimental_connect_to_cluster(resolver)\n',
    '    tf.tpu.experimental.initialize_tpu_system(resolver)\n',
    '    strategy = tf.distribute.TPUStrategy(resolver)\n',
    '    print(f"TPU available: {resolver.cluster_spec().as_dict()}")\n',
    '    print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")\n',
    '    BATCH_SIZE = 128\n',
    'except (ValueError, tf.errors.NotFoundError):\n',
    '    print("No TPU found. Using GPU/CPU.")\n',
    '    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices("GPU")) > 1 else tf.distribute.get_strategy()\n',
    '    print(f"Devices: {tf.config.list_physical_devices()}")\n',
    '    BATCH_SIZE = 32\n',
    '    import subprocess\n',
    '    subprocess.run(["nvidia-smi"], shell=True)\n',
]

# Cell 4: Data loading - remove .cache() for TPU compat, use BATCH_SIZE from strategy
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
    '\n',
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
    'print(f"Batch size: {BATCH_SIZE}")\n',
    '\n',
    'AUTOTUNE = tf.data.AUTOTUNE\n',
    'train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)\n',
    'val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)\n',
]

# Cell 8 (Run 4 viz cell): Also need to fix the f-string for f"Run {m['run']}:..."
# Actually that's in cell 9 which has the save code - let me check which cell index
# Let me just skip the broken cells by replacing them

# Cell 8: Run 4 classification report + confusion matrix (index 8)
# The bug is `f"Run {m['run']}:..."` in f-strings - need to change m['run'] to m.get('run')
# Let me fix all such issues

for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        # Fix f-string issues with nested single quotes in dict access
        fix_count = 0
        if "m['run']" in src:
            src = src.replace("m['run']", "m.get('run')")
            fix_count += 1
        if "m['best_val_acc']" in src:
            src = src.replace("m['best_val_acc']", "m.get('best_val_acc')")
            fix_count += 1
        if "m['best']" in src:
            src = src.replace("m['best']", "m.get('best')")
            fix_count += 1
        if "m['epoch']" in src:
            src = src.replace("m['epoch']", "m.get('epoch')")
            fix_count += 1
        if "m['final_val']" in src:
            src = src.replace("m['final_val']", "m.get('final_val')")
            fix_count += 1
        if "m['final_train']" in src:
            src = src.replace("m['final_train']", "m.get('final_train')")
            fix_count += 1
        if "m['epochs']" in src:
            src = src.replace("m['epochs']", "m.get('epochs')")
            fix_count += 1
        if "m['final_val_acc']" in src:
            src = src.replace("m['final_val_acc']", "m.get('final_val_acc')")
            fix_count += 1
        if "m['final_train_acc']" in src:
            src = src.replace("m['final_train_acc']", "m.get('final_train_acc')")
            fix_count += 1
        if "m['epochs_run']" in src:
            src = src.replace("m['epochs_run']", "m.get('epochs_run')")
            fix_count += 1
        if "m['best_epoch']" in src:
            src = src.replace("m['best_epoch']", "m.get('best_epoch')")
            fix_count += 1
        if fix_count > 0:
            c["source"] = [src]
            print(f"Cell {i}: Fixed {fix_count} f-string issues")

# Cell 14 (Run 5 training): Wrap model creation in strategy.scope()
cell14 = cells[14]
src14 = "".join(cell14["source"])
if "strategy.scope" not in src14:
    src14 = src14.replace(
        "m = create_model_v5(NUM_CLASSES)",
        "with strategy.scope():\n            m = create_model_v5(NUM_CLASSES)"
    )
    # Fix the first occurrence in the model definition cell too
    cell14["source"] = [src14]
    print("Cell 14: Added strategy.scope() wrapping")

# Also need to make sure the model definition + compile at top of Run 5 uses strategy
# That's in cell 12 (create_model_v5 function definition + compile)
cell12 = cells[12]
src12 = "".join(cell12["source"])
if "strategy.scope" not in src12:
    src12 = src12.replace(
        "model_v5 = create_model_v5(NUM_CLASSES)",
        "with strategy.scope():\n            model_v5 = create_model_v5(NUM_CLASSES)"
    )
    cell12["source"] = [src12]
    print("Cell 12: Added strategy.scope() wrapping")

# Update Run 5 markdown (cell 10) to reflect current settings
cells[10]["source"] = [
    "## RUN 5: IMPROVED ARCHITECTURE TARGETING 93%\n",
    "### Changes from Run 4:\n",
    "- 2 Conv layers per block (deeper): 32→64→128→256\n",
    "- Double Dense head: 512 → 256 → 9\n",
    "- SGD with momentum (0.9) + CosineDecay LR (0.01 → 1e-5)\n",
    "- Label smoothing (0.1)\n",
    "- More aggressive augmentation (+Translation)\n",
    "- TPU-optimized: 5 sub-runs x 30 epochs, batch_size=128 on TPU\n",
    "- Ensemble top 3 runs\n"
]

# Save
with open("aie231assgnmnt3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

if os.path.exists("run5_progress.json"):
    os.remove("run5_progress.json")

print("\nNotebook updated for TPU + GPU compatibility")
print(f"Total cells: {len(cells)}")
