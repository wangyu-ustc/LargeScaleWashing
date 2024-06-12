import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class MQUAKEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        
        data_dir = Path(data_dir)
        cf_loc = data_dir / "MQuAKE-CF.json"

        with open(cf_loc, "r") as f:
            self.data = json.load(f)

        self.data = [x for x in self.data if len(x['requested_rewrite']) > 1]
        split = int(len(self.data) * 0.8)
        self.data = self.data[:split]

        new_data = []
        count = 1
        for item in self.data:
            for x in item['requested_rewrite']:
                new_data.append({
                    'case_id': count,
                    'requested_rewrite': x
                })
                count += 1

        self.data = new_data
        if size is not None:
            self.data = self.data[:size]
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiMQUAKEDataset(MQUAKEDataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)
