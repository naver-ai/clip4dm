import copy
import json
from collections import namedtuple
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils import *

output = namedtuple(
    "output",
    ["idx", "img", "caption", "foil", "img_path", "groupby", "score", "references"],
)


class FOILDataset:
    def __init__(self, data_path, img_dir, reference_path=None):
        with open(data_path) as f:
            self.data = json.load(f)
        self.id2filename = {i["id"]: i["file_name"] for i in self.data["images"]}
        if reference_path:
            self.references = pd.DataFrame(json.loads(open(reference_path).read())["annotations"])
        self.data = self.data["annotations"]
        self.img_dir = img_dir

    def get_image_from_id(self, image_id):
        img_path = self.id2filename[image_id]
        img = Image.open(f"{self.img_dir}/{img_path}")
        return img

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = self.get_image_from_id(sample["image_id"])
        caption = sample["caption"].strip()
        foil_word = sample["foil_word"].strip()
        _id = sample["id"]
        image_id = sample["image_id"]
        refs = self.references[self.references.image_id == image_id]
        refs = refs[refs.id != _id].caption.tolist()
        return output(
            idx=_id,
            img=img,
            caption=caption,
            references=refs,
            foil=foil_word,
            img_path=image_id,
            groupby=None,
            score=None,
        )

    def __len__(self):
        return len(self.data)


class NoCapsDataset:
    def __init__(self, data_path, img_dir):
        self.data = []
        self.img_dir = img_dir
        with open(data_path) as f:
            data = json.load(f)
        for _idx, sample in enumerate(data):
            sample["id"] = _idx
            self.data.append(sample)
            # once with foil and once with original
            orig_sample = copy.deepcopy(sample)
            orig_sample["foil"] = orig_sample["baseline"]  # original로 교
            orig_sample["replacement"] = ["", ""]
            self.data.append(orig_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(f"{self.img_dir}/{sample['image_path']}")
        caption = sample["foil"].strip()
        foil = sample["replacement"][1]
        references = sample["references"]
        if sample["replacement"][1] == sample["replacement"][0]:
            foil = "orig"
        return output(
            idx=sample["id"],
            img=img,
            caption=caption,
            references=references,
            foil=foil,
            img_path=sample["image_path"],
            groupby=sample["domain"],
            score=None,
        )


class HATDataset:
    def __init__(self, filepath, img_dir):
        with open(filepath, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["image_path"]
        img = Image.open(f"{self.img_dir}/{img_path}")
        caption = sample["caption"]
        references = sample["references"]
        token_align = sample["grounding"]
        foil_words = [token.replace(".", "").replace(",", "") for token, align in token_align if align]
        foil_word = " ".join(foil_words)
        if foil_words == "":
            foil_word = "orig"
        return output(
            idx=sample["sample_id"],
            img=img,
            caption=caption,
            references=references,
            foil=foil_word,
            img_path=img_path,
            groupby=None,
            score=None,
        )


class RichHFDataset:
    def __init__(self, data_path, img_dir):
        self.data = []
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")  # we do not need gpu for loading dataset
        for sample in list(tf.data.TFRecordDataset(data_path)):
            example = tf.train.Example()
            example.ParseFromString(sample.numpy())
            feat_map = example.features.feature
            filename = feat_map["filename"].bytes_list.value[0].decode()
            score = feat_map["misalignment_score"].float_list.value[0]
            token_label = feat_map["prompt_misalignment_label"].bytes_list.value[0]
            token_label = token_label.decode()
            img_path = f"{img_dir}/{Path(filename).stem}.png"
            caption = open(f"{img_dir}/{Path(filename).stem}.txt", "r").read()
            tokens = richhf_tokenize(caption)
            tokens = [t for t in tokens if t]
            token_label = token_label.split()
            token_align = list(zip(tokens, token_label))
            foil_words = [f"{_idx}_{token}" for _idx, (token, align) in enumerate(token_align) if align == "0"]
            self.data.append(
                {
                    "img_path": img_path,
                    "caption": caption,
                    "foil_word": " ".join(foil_words),
                    "filename": filename,
                    "score": score,
                }
            )

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["img_path"]
        img = Image.open(img_path)
        return output(
            idx=idx,
            img=img,
            caption=sample["caption"],
            references=[],
            foil=sample["foil_word"],
            img_path=sample["filename"],
            groupby=None,
            score=sample["score"],
        )

    def __len__(self):
        return len(self.data)


class SeeTrueDataset:
    def __init__(self, data_path=None):
        from datasets import load_dataset

        try:
            self.data = load_dataset("mismatch-quest/SeeTRUE-Feedback")["test"]
        except:
            raise "download PIL==9.4.0"

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["image"]
        caption = sample["image_caption"]
        foil_word = sample["caption_misalignment"]
        image_id = f"{sample['id_in_source_dataset']}"
        return output(
            idx=idx,
            img=img,
            caption=caption,
            references=[],
            foil=foil_word,
            img_path=image_id,
            groupby=sample["dataset_source"],
            score=None,
        )

    def __len__(self):
        return len(self.data)
