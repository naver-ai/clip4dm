import argparse
import json
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm


def download(args):
    with open(args.foil_path, "r") as f:
        nocaps_foil = json.load(f)

    with open(args.meta_path, "r") as f:
        nocaps_meta = json.load(f)
    images = nocaps_meta["images"]

    images_dict = {_image["file_name"]: _image["coco_url"] for _image in images}

    for _data in tqdm(nocaps_foil):
        img_path = _data["image_path"]
        img_url = images_dict[img_path]
        img = requests.get(img_url).content
        img = Image.open(BytesIO(img))
        img.save(f"{args.save_dir}/{img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--foil_path", type=str, default="./data/nocaps-val-foil.json")
    parser.add_argument("--meta_path", type=str, default="./data/nocaps_val_4500_captions.json")
    parser.add_argument("--save_dir", type=str, default="./data/nocaps")
    args = parser.parse_args()
    download(args)
