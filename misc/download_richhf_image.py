import argparse
import io
from pathlib import Path

import tensorflow as tf
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download(args):
    ds = load_dataset("yuvalkirstain/pickapic_v1")  # it takes few hours

    full_image = []
    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        feat_map = example.features.feature

        # original filename which can be mapped to images in pick-a-pic dataset.
        filename = feat_map["filename"].bytes_list.value[0].decode()
        full_image.append(filename)

    full_image = {Path(_).stem: _ for _ in full_image}
    for sample in tqdm(ds["test"]):
        for jpg_key in ["0", "1"]:
            if sample[f"image_{jpg_key}_uid"] in full_image:
                image = Image.open(io.BytesIO(sample[f"jpg_{jpg_key}"]))
                image.save(f"{args.save_dir}/{sample[f'image_{jpg_key}_uid']}.png")
                open(f"{args.save_dir}/{sample[f'image_{jpg_key}_uid']}.txt", "w").write(sample["caption"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--foil_path", type=str, default="./data/richhf/test.tfrecord")
    parser.add_argument("--save_dir", type=str, default="./data/richhf/test")
    args = parser.parse_args()
    download(args)
