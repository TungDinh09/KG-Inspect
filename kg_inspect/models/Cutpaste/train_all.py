import os
import gc
import paddle

# Define your list of defect types
defect_types = [ 'pcb4', 'cashew', 'pcb1', 'pcb2', 'pcb3', 'capsules', 'candle', 'fryum', 'macaroni1', 'pipe_fryum', 'macaroni2', 'chewinggum']
# Base training command template
# You can adjust parameters here for your system
BASE_CMD = (
    "python train.py "
    "--data_dir /home/guest/Documents/TUNG/KG-Inspect/data/visa-data "  # ← note the space here
    "--type {defect} "
    "--variant 3way "
    "--batch_size 64 "
    "--epochs 256 "
    "--cuda True "
)


for defect in defect_types:
    print(f"\n🚀 Starting training for: {defect}\n{'='*60}")
    os.system(BASE_CMD.format(defect=defect))

    # --- Memory cleanup between runs ---
    print(f"🧹 Cleaning GPU memory after {defect}")
    paddle.device.cuda.empty_cache()
    gc.collect()

    print(f"✅ Finished training for: {defect}\n{'='*60}\n")
