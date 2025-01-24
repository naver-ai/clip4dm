import numpy as np
import open_clip
import torch
from open_clip.tokenizer import SimpleTokenizer
from transformers import AutoModel

from src.utils import space_tokenize, tokenizer2space

device = "cuda"


backbone2pretrained = {
    "ViT-B-32": "openai",
    "ViT-L-14": "openai",
    "ViT-H-14": "laion2b_s32b_b79k",
}
backbone2hf = {
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-L-14": "openai/clip-vit-large-patch14",
    "ViT-L-14-336": "openai/clip-vit-large-patch14-336",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


def load_model(backbone_name, pretrained=None):
    """
    load model from backbone and pratrained.
    Args:
        backbone_name (str)
        pretrained (str)
    Returns:
        model, preprocess, tokenizer
    """
    if pretrained is None:
        pretrained = backbone2pretrained[backbone_name]
    tokenizer = SimpleTokenizer()
    if "openai" in pretrained:
        pretrained = backbone2hf[backbone_name]
    elif "/" not in pretrained:
        pretrained = f"laion/CLIP-{backbone_name}-{pretrained.replace('_', '-')}"
    _, _, preprocess = open_clip.create_model_and_transforms(
        backbone_name
    )  # we observed hf ImageProcessor slightly differs from open_clip processor
    model = AutoModel.from_pretrained(pretrained)
    model.to(device)
    return model, preprocess, tokenizer


class CLIP4DM:
    def __init__(
        self,
        backbone_name,
        pretrained=None,
        template="A photo depicts ",
        start_layer_text=-3,
        sorted_by="most_negative",
    ):
        model, preprocess, tokenizer = load_model(backbone_name, pretrained)
        self.model = model.to(torch.float16)
        self.template = template
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.start_layer_text = start_layer_text
        self.sorted_by = sorted_by
        self.context_length = self.model.text_model.embeddings.position_embedding.weight.shape[0]

    def get_attribution_map(self, image, texts, model, device):
        model_args = {}
        model_args["pixel_values"] = self.preprocess(image).unsqueeze(0).to(device).to(torch.float16)
        model_args["input_ids"] = self.tokenizer(texts).to(device)
        outputs = model(**model_args, output_attentions=True, output_hidden_states=True)
        logit = outputs.logits_per_text / model.logit_scale.exp()

        batch_size = 1
        text_attn_blocks = outputs.text_model_output.attentions
        text_hidden_blocks = outputs.text_model_output.hidden_states

        if self.start_layer_text < 0:
            text_attn_blocks = text_attn_blocks[self.start_layer_text :]
            text_hidden_blocks = text_hidden_blocks[self.start_layer_text :]

        model.zero_grad()
        num_tokens = text_attn_blocks[0].shape[-1]
        R_text = torch.zeros(num_tokens, num_tokens, dtype=text_attn_blocks[0].dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        one_hot = np.zeros((logit.shape[0], logit.shape[0]), dtype=np.float32)
        one_hot[torch.arange(logit.shape[0]), [0]] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        logit = torch.sum(one_hot.to(device) * logit)
        for attn in text_attn_blocks[::-1]:
            grad = torch.autograd.grad(logit, [attn], retain_graph=True)[0].detach()
            cam = attn.detach()  # 1, n_head, seq_len, seq_len
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = grad * cam  # gradient of attention map * attention map
            cam = cam.mean(dim=1)
            R_text = R_text + cam
        R_text = R_text / len(text_attn_blocks)
        return R_text, logit

    def __call__(self, img, text, threshold=-5e-5):
        text = self.template + text
        R_text, logit = self.get_attribution_map(model=self.model, image=img, texts=text, device=device)
        text_encoding = self.tokenizer(text)[0]
        R_text = R_text[0]
        CLS_idx = text_encoding.argmax(dim=-1)
        r_i = R_text[CLS_idx, 1:CLS_idx]
        text_scores = r_i.flatten()
        text_tokens = self.tokenizer.encode(text)
        text_tokens_decoded = [self.tokenizer.decode([a]) for a in text_tokens]
        if not len(text_tokens_decoded) == len(text_scores):  # ignore tokens over length 77
            print("length over 77 is ignored", len(text_tokens_decoded))
        negative_word = [(txt, score.item()) for txt, score in zip(text_tokens_decoded, text_scores)]
        space_token = space_tokenize(text)
        space2token = tokenizer2space(tokenizer=self.tokenizer, text=text)
        bpe_token_idx = 0
        aggr_token = []
        for (
            space_token_idx,
            bpe_token,
        ) in space2token.items():  # aggregate tokens to word based on space
            bpe_token_score = [_score for _, _score in negative_word[bpe_token_idx : bpe_token_idx + len(bpe_token)]]
            bpe_token_idx += len(bpe_token)
            _space_token = space_token[space_token_idx]
            try:
                aggr_token.append((_space_token, sum(bpe_token_score) / len(bpe_token_score)))
            except ZeroDivisionError:
                print("seq over 77 is ignored.")

        text_scores = aggr_token
        text_scores = text_scores[len(self.template.split()) :]
        negative_score_list = []
        if text_scores:
            if self.sorted_by == "threshold":  # For RichHF, HAT, SeeTrue
                text_scores = [(word, score) for word, score in text_scores if word not in '.,!"?;:" ']
                text_scores = [(word, score) for word, score in text_scores if word.strip()]
                all_score = text_scores[:]
                text_scores = list(enumerate(text_scores))  # we need word index to merge adjacent words
                text_scores = [(f"{_idx}_{word}", score) for _idx, (word, score) in text_scores if score < threshold]
                text_scores = sorted(text_scores, key=lambda e: e[1])
                if text_scores:
                    negative_score_list = [_score for _word, _score in text_scores]
                    negative_score = sum(negative_score_list)
                else:
                    negative_score = 0
                negative_word = " ".join([word for word, _ in text_scores])
            elif self.sorted_by == "most_negative":
                sorted_text = sorted(text_scores, key=lambda e: e[1])  # sort in ascending order and get the first one
                negative_word, negative_score = sorted_text[0]
                all_score = text_scores[:]
        return {
            "negative_word": negative_word,
            "sum_negative_attribution": negative_score,
            "negative_word_score_list": negative_score_list,
            "clip_score": max(logit.item(), 0) * 2.5,  # CLIPScore
            "word_attributions": all_score,
        }
