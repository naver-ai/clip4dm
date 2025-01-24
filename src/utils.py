import json
import re


def get_reference_term(interpreter, candidate, references, template):
    tokenizer = interpreter.tokenizer
    model = interpreter.model
    caption_embeds = model.text_projection(model.text_model(tokenizer(template + candidate).to("cuda"))[1])
    caption_embeds = caption_embeds / caption_embeds.norm(p=2, dim=-1, keepdim=True)
    r_c_score = 0
    for _ref in references:
        # https://github.com/huggingface/transformers/blob/v4.43.2/src/transformers/models/clip/modeling_clip.py#L1402
        pool_output = model.text_model(tokenizer(template + _ref).to("cuda"))[1]
        ref_embeds = model.text_projection(pool_output)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)
        _r_c_score = (caption_embeds @ ref_embeds.T).item()
        if _r_c_score > r_c_score:
            r_c_score = _r_c_score
        return r_c_score * 2.5


# https://github.com/google-research/google-research/blob/master/richhf_18k/match_label_to_token.py
def richhf_tokenize(caption):
    delimiters = ',.?!":; '
    pattern = "|".join(map(re.escape, delimiters))
    # Split by punctuation or space and remove empty tokens.
    tokens = re.split(pattern, caption)
    tokens = [t for t in tokens if t]
    return tokens


def word_seq_to_tokens_with_score(negative_word, word_score):
    if not negative_word or not word_score:
        return []
    parts = negative_word.split(" ")
    parts_score = list(zip(parts, word_score))
    parts_score = sorted(parts_score, key=lambda e: int(e[0].split("_")[0]))
    result = []
    current_word = []
    current_scores = []
    last_index = -1
    for part, score in parts_score:
        index, word = part.split("_")
        index = int(index)
        if index == last_index + 1:
            current_word.append(word)
            current_scores.append(score)
        else:
            if current_word:
                avg_score = sum(current_scores) / len(current_scores)
                result.append((" ".join(current_word), avg_score))
            current_word = [word]
            current_scores = [score]
        last_index = index
    if current_word:
        avg_score = sum(current_scores) / len(current_scores)
        result.append((" ".join(current_word), avg_score))
    return result


def word_seq_to_tokens(negative_word):
    # "7_african 13_closed" -> ["african", "closed"]
    # "7_african 8_word 13_closed" -> ["african word", "closed"]
    if not negative_word:
        return ""
    parts = negative_word.split(" ")
    parts = sorted(parts, key=lambda e: int(e.split("_")[0]))
    result = []
    current_word = []
    last_index = -1
    for part in parts:
        index, word = part.split("_")
        index = int(index)
        if index == last_index + 1:
            current_word.append(word)
        else:
            if current_word:
                result.append(" ".join(current_word))
            current_word = [word]
        last_index = index
    if current_word:
        result.append(" ".join(current_word))
    return result


def load_jsonl(jsonl):
    with open(jsonl, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    return samples


def check_different_args(parser, args):
    different_args = {}
    for action in parser._actions:
        if action.dest != "help":  
            arg_name = action.dest
            default_value = action.default
            if args not in ["backbone", "data_path"]:
                current_value = getattr(args, arg_name)
                if current_value != default_value:
                    different_args[arg_name] = current_value
    return different_args


def get_name_from_args(parser, args):
    diff_args = check_different_args(parser, args)
    names = []
    for key, value in diff_args.items():
        if key == "data_path":
            continue
        if isinstance(value, str):
            names.append(value)
        else:
            names.append(f"{key}_{value}")
    return "_".join(names)


def metrics_for_multiprediction(ground_truth, predicted):
    TP = FP = FN = 0
    for item in ground_truth:
        if item in predicted:
            TP += 1
            predicted.remove(item)
        else:
            FN += 1
    FP = len(predicted)
    return TP, FP, FN


def space_tokenize(text):
    tokens = []
    token = ""
    for char in text:
        if char == " ":
            if token:
                tokens.append(token)
                token = ""
        elif char in [".", ","]:
            if token:
                tokens.append(token)
                token = ""
            tokens.append(char)
        else:
            token += char
    if token:
        tokens.append(token)
    return tokens


def tokenizer2space(tokenizer, text):  # only works for bpe (bbpe X)
    space_tokens = space_tokenize(text)
    encoded = tokenizer.encode(text)
    result = {}
    encoded_idx = 0
    for idx, token in enumerate(space_tokens):
        token_encoded = tokenizer.encode(token)
        result[idx] = encoded[encoded_idx : encoded_idx + len(token_encoded)]
        encoded_idx += len(token_encoded)
    return result
