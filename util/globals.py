from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR, PRED_DIR, EDIT_NONE_KV_DIR, EDIT_NONE_KV_EXPAND_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
        data["PRED_DIR"],
        data["EDIT_NONE_KV_DIR"],
        data['EDIT_NONE_KV_EXPAND_DIR']
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
