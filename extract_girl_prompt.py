import markovify
import pandas as pd
import requests
import io
import pickle

from transformers import CLIPTokenizer

LIMIT_CLIP_TEXT_ENC=75
girl_dictionary = ["girl", "woman", "female", "princess",
              "actless", "maid", "gal", "maiden", "waifu",
              "wife", "loli", "kawaii"]
boy_dictionary = ["boy", "man", "male", "guy",
                  "dude", "geezer", "chap", "fellow","bloke"]

response=requests.get("https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet")
inmemory_file=io.BytesIO(response.content)
with io.BytesIO(response.content) as inmemory_file:
    df = pd.read_parquet(inmemory_file)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

prompts=[]
for idx,row in df.iterrows():
    prompt=row["prompt"]

    tokens = tokenizer.tokenize(prompt)

    if (len(tokens) > LIMIT_CLIP_TEXT_ENC):
        continue

    if (any([x in prompt for x in girl_dictionary])):
        if (all([not x in prompt for x in boy_dictionary])):
            prompts.append(prompt)

print(len(prompts))

text_model = markovify.Text(prompts)

model_json = text_model.to_json()
with open(f"extract_prompts_{len(prompts)}.pickle", "wb") as f:
    pickle.dump(prompts, f, pickle.HIGHEST_PROTOCOL)