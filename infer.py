from PIL import Image

from src.clip4dm import CLIP4DM

if __name__ == "__main__":
    clip4dm = CLIP4DM(backbone_name="ViT-H-14", sorted_by="threshold")
    img = Image.open("misc/a_cup_of_coffee.jpg")

    text = "a cup of coke"
    print(text)
    print(clip4dm(img=img, text=text))
    # {'negative_word': '3_coke', 'sum_negative_attribution': -0.0004444122314453125, 'negative_word_score_list': [-0.0004444122314453125], 'clip_score': 0.52459716796875, 'word_attributions': [('a', -3.6835670471191406e-05), ('cup', 0.0002104043960571289), ('of', 5.716085433959961e-05), ('coke', -0.0004444122314453125)]}

    text = "a cup of coffee"
    print(text)
    print(clip4dm(img=img, text=text))
    # {'negative_word': '', 'sum_negative_attribution': 0, 'negative_word_score_list': [], 'clip_score': 0.6884765625, 'word_attributions': [('a', -3.0338764190673828e-05), ('cup', 4.750490188598633e-05), ('of', 1.7285346984863281e-06), ('coffee', 0.00013780593872070312)]}
