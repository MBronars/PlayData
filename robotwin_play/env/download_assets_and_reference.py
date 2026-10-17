import os, subprocess
from huggingface_hub import hf_hub_download
ROOT = "/data/user_data/mbronars/packages/RoboTwin/assets"
for f in ["embodiments.zip", "objects.zip", "background_texture.zip"]:
    p = hf_hub_download("TianxingChen/RoboTwin2.0", f, repo_type="dataset", local_dir=ROOT)
    subprocess.run(["unzip", "-q", "-o", p, "-d", ROOT], check=True)
    os.remove(p)
    print("assets", f, flush=True)
REF = "/data/user_data/mbronars/robotwin_ref"
tasks = ["place_object_basket", "stack_blocks_two", "open_microwave", "put_object_cabinet", "move_can_pot",
         "place_container_plate", "stack_bowls_two", "lift_pot", "place_a2b_left", "turn_switch", "open_laptop"]
for emb in ["aloha-agilex_clean_50", "franka_clean_50"]:
    for t in tasks:
        f = f"dataset/{t}/{emb}.zip"
        out = f"{REF}/{t}/{emb}"
        if os.path.exists(out):
            continue
        try:
            p = hf_hub_download("TianxingChen/RoboTwin2.0", f, repo_type="dataset", local_dir=REF + "/_dl")
        except Exception as e:
            print("missing", f, e, flush=True); continue
        os.makedirs(out, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", p, "-d", out], check=True)
        os.remove(p)
        print("ref", t, emb, flush=True)
print("DOWNLOAD_DONE")
