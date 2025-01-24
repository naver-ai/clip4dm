import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.clip4dm import CLIP4DM, backbone2pretrained
from src.dataset import FOILDataset, HATDataset, NoCapsDataset, RichHFDataset, SeeTrueDataset
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(df, foil_is_positive=True):
    df["GT_foil_word"] = df.GT_foil_word.apply(lambda foil: foil if foil.strip() else "orig")
    df["is_foil"] = df.GT_foil_word != "orig"
    df["is_orig"] = df.GT_foil_word == "orig"
    LA = df.is_correct.sum() / df.is_foil.sum()
    NLI = None
    AP_refclipscore = None
    df["negative_pair_score"] = 1 - df["pair_score"] / 2.5
    df["FCLIPScore"] = df.apply(lambda row: row.sum_negative_attribution * row.negative_pair_score, axis=1)
    if "nli_score" in df:
        NLI = df["nli_score"].sum() / df.shape[0]
    if foil_is_positive:  # FOIL: 0 // original: 1
        AP_clipscore = average_precision_score(df["is_orig"], df["pair_score"])
        if "ref_clipscore" in df:
            AP_refclipscore = average_precision_score(df["is_orig"], df["ref_clipscore"])
        AP_FCLIPScore = average_precision_score(df["is_orig"], df["FCLIPScore"])
    else:  # FOIL: 1 // original : 0
        AP_clipscore = average_precision_score(df["is_foil"], -df["pair_score"])
        if "ref_clipscore" in df:
            AP_refclipscore = average_precision_score(df["is_foil"], -df["ref_clipscore"])
        AP_FCLIPScore = average_precision_score(df["is_foil"], -df["FCLIPScore"])
    return {
        "AP_clipscore": AP_clipscore,
        "AP_refclipscore": AP_refclipscore,
        "AP_FCLIPScore": AP_FCLIPScore,
        "LA": LA,
        "NLI": NLI,
    }


def main(args):
    if any(d in args.data_path for d in ["hat", "richhf", "seetrue"]):
        args.sorted_by = "threshold"
        print(f"{args.data_path} needs to be load from threshold")
    clip4dm = CLIP4DM(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        template=args.template,
        start_layer_text=args.start_layer_text,
        sorted_by=args.sorted_by,
    )
    confusion_matrix = defaultdict(int)
    correct = 0
    nli_score, _nli_score = 0, None
    if args.pretrained is None:
        pretrained = f"{args.backbone}/{backbone2pretrained[args.backbone]}"
    else:
        pretrained = f"{args.backbone}/{args.pretrained}"
    pretrained = pretrained.replace("/", "-")
    save_path = f"{args.output}/{Path(args.data_path).stem}/{pretrained}/L_{args.start_layer_text}"
    save_path += f"/{get_name_from_args(parser, args)}"
    os.makedirs(save_path, exist_ok=True)
    f = open(f"{save_path}/samples.jsonl", "w", buffering=1)
    if "seetrue" in args.data_path.lower():
        data = SeeTrueDataset(args.data_path)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        nli_model.to(device)
        nli_score = 0
    elif "nocaps" in args.data_path.lower():
        data = NoCapsDataset(args.data_path, args.img_dir)
    elif "richhf" in args.data_path.lower():
        data = RichHFDataset(args.data_path, args.img_dir)
    elif "hat" in args.data_path.lower():
        data = HATDataset(args.data_path, args.img_dir)
    elif "foil" in args.data_path.lower():
        data = FOILDataset(args.data_path)

    # for logging frequent words
    word_dict = defaultdict(int)
    for _idx in tqdm(range(len(data))):
        sample = data[_idx]
        _id, img, caption, foil_word = (
            sample.idx,
            sample.img,
            sample.caption,
            sample.foil,
        )
        image_id, groupby, gt_score = sample.img_path, sample.groupby, sample.score
        references = sample.references
        ref_score = None
        if references and args.get_refclipscore:
            ref_score = get_reference_term(clip4dm, caption, references, args.template)
        result = clip4dm(img, caption, threshold=args.epsilon)
        sum_negative_attribution = result["sum_negative_attribution"]
        negative_word = result["negative_word"]
        score = result["clip_score"]
        negative_word = negative_word.strip().lower()
        foil_word = foil_word.strip().lower()
        foil_is_positive = True  # if foil is 1
        if "richhf" in args.data_path.lower():
            foil_words = foil_word.split()
            negative_words = negative_word.split()
            TP, FP, FN = metrics_for_multiprediction(foil_words, negative_words)
            TN = len(richhf_tokenize(caption)) - (TP + FP + FN)
            confusion_matrix["TP"] += TP
            confusion_matrix["FP"] += FP
            confusion_matrix["FN"] += FN
            confusion_matrix["TN"] += TN
            if TP > 0:
                is_correct = True
                correct += 1
            else:
                is_correct = False
            for _negative_word in negative_word.split():
                if "_" in _negative_word:
                    _negative_word = _negative_word.split("_")[1]
                word_dict[_negative_word] += 1
            foil_is_positive = False
        elif "hat" in args.data_path.lower():
            is_correct = False
            if negative_word:
                negative_words = word_seq_to_tokens_with_score(negative_word, result["negative_word_score_list"])
                negative_word, sum_negative_attribution = sorted(negative_words, key=lambda e: e[1])[0]
                if negative_word in foil_word:
                    is_correct = True
                    correct += 1
                else:
                    is_correct = False
            else:
                is_correct = False
            foil_is_positive = False
        elif "seetrue" in args.data_path.lower():
            negative_word = word_seq_to_tokens(negative_word)
            is_correct = False
            for word in negative_word:
                for _word in word.split():
                    if _word in foil_word:  # just for logging
                        correct += 1
                        is_correct = True
            negative_word = ", ".join(negative_word)
        else:  # FOIL, nocaps-FOIL
            negative_word = negative_word.split()[0].split("_")[-1]
            word_dict[negative_word] += 1
            if negative_word == foil_word:
                is_correct = True
                correct += 1
            else:
                is_correct = False
        if "seetrue" in args.data_path.lower():
            if negative_word:
                x = nli_tokenizer.encode(
                    foil_word,
                    negative_word,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                )
                logits = nli_model(x.to(device))[0]
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                _nli_score = probs[:, 1].item()
            else:  # if word is blank, set _nli_score as 0
                _nli_score = 0
            nli_score += _nli_score
        output_result = {
            "id": _id,
            "caption": caption,
            "pair_score": score,  # CLIP logit
            "GT_foil_word": foil_word,  # ground truth negative_word
            "predicted_foil_word": negative_word,  # predicted foil word
            "sum_negative_attribution": sum_negative_attribution,  # sum(r_i if r_i < threshold) or min(r_i)
            "gt_score": gt_score,  # (for Rich-HF) GT alignment score
            "is_correct": is_correct,  # (for FOIL) foil_word == negative_word or foil_word in negative_word
            "image_id": image_id,  # image id
            "word_attributions": result["word_attributions"],  # all r_i
        }
        if ref_score:
            output_result["ref_clipscore"] = 2 * score * ref_score / (score + ref_score)
        if _nli_score:  # NLI score (for seetrue dataset)
            output_result["nli_score"] = _nli_score
        if groupby is not None:  # e.g. image source, data domain
            output_result["groupby"] = groupby
        if gt_score is not None:
            output_result["gt_score"] = gt_score
        f.write(json.dumps(output_result) + "\n")
    f.close()
    output = load_jsonl(f"{save_path}/samples.jsonl")
    output = pd.DataFrame(output)
    metrics = calculate_metrics(output, foil_is_positive=foil_is_positive)

    confusion_matrix = {}
    TP = confusion_matrix.pop("TP", 0)
    FP = confusion_matrix.pop("FP", 0)
    FN = confusion_matrix.pop("FN", 0)
    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        confusion_matrix["precision"] = precision
        confusion_matrix["recall"] = recall
        confusion_matrix["f1"] = 2 * (precision * recall) / (precision + recall)
    pearson = spearman = None
    if "richhf" in args.data_path.lower():
        from scipy.stats import pearsonr, spearmanr

        pearson = pearsonr(output["gt_score"], output["pair_score"]).statistic
        spearman = spearmanr(output["gt_score"], output["pair_score"]).statistic
        pearson_fclipscore = pearsonr(output["gt_score"], output["FCLIPScore"]).statistic
        spearman_fclipscore = spearmanr(output["gt_score"], output["FCLIPScore"]).statistic

    # save output result as text file
    output_json = {}
    output_json["args"] = args.__dict__
    output_json["LA"] = metrics["LA"]
    output_json["AP"] = {
        "F-CLIPScore": metrics["AP_FCLIPScore"],
        "CLIPScore": metrics["AP_clipscore"],
    }
    if metrics["AP_refclipscore"]:
        output_json[f"AP"]["refCLIPScore"] = metrics["AP_refclipscore"]
    if metrics["NLI"]:
        output_json["NLI"] = metrics["NLI"]
    if confusion_matrix:
        output_json["confusion_matrix"] = confusion_matrix
    if pearson:
        output_json["pearson"] = {"F-CLIPScore": pearson_fclipscore, "CLIPSCore": pearson}
        output_json["spearman"] = {
            "F-CLIPScore": spearman_fclipscore,
            "CLIPSCore": spearman,
        }

    if "groupby" in output:
        groupby_output = output.groupby("groupby")
        for group, _df in groupby_output:
            metrics = calculate_metrics(_df)
            output_json[f"LA/{group}"] = metrics["LA"]
            output_json[f"AP/{group}"] = {
                "F-CLIPScore": metrics["AP_FCLIPScore"],
                "CLIPScore": metrics["AP_clipscore"],
            }
            if metrics["NLI"]:
                output_json[f"NLI/{group}"] = metrics["NLI"]
            if metrics["AP_refclipscore"]:
                output_json[f"AP/{group}"]["refCLIPScore"] = metrics["AP_refclipscore"]
    print(output_json)
    with open(f"{save_path}/result.json", "w") as f:
        json.dump(output_json, f)
    print(f"output is saved in {save_path}")


if __name__ == "__main__":

    def str2bool(value):
        if isinstance(value, str):
            _true_set = {"yes", "true", "t", "y", "1"}
            _false_set = {"no", "false", "f", "n", "0"}
            value = value.lower()
            if value in _true_set:
                return True
            if value in _false_set:
                return False

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/nocaps_val.json", help="data json file path")
    parser.add_argument("--img_dir", type=str, default="./data/nocaps", help="saved img directory")
    parser.add_argument(
        "--start_layer_text",
        type=int,
        default=-3,
        help="start layer for text transformer",
    )
    parser.add_argument("--output", type=str, default="./output", help="output path")
    parser.add_argument(
        "--template",
        type=str,
        default="A photo depicts ",
        help="template prompt used in text encoder",
    )  # follows CLIPScore
    parser.add_argument("--backbone", type=str, default="ViT-B-32", help="backbone name")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained name")
    parser.add_argument(
        "--sorted_by",
        type=str,
        default="threshold",
        choices=["most_negative", "threshold"],
        help="head aggregation method",
    )  # most_negative: foil, nocaps-foil // threshold: hat, seetrue, richhf
    parser.add_argument("--epsilon", type=float, default=-5e-5, help="when using threshold")
    parser.add_argument(
        "--get_refclipscore",
        type=str2bool,
        default=False,
        help="get reference term for refclipscore",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="for FOIL, `coco14/captions_val2014.json` path to get ref_CLIPScore",
    )
    args = parser.parse_args()
    main(args)
