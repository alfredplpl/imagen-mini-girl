import markovify
import pandas as pd
import requests
import io

response=requests.get("https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet")
inmemory_file=io.BytesIO(response.content)
with io.BytesIO(response.content) as inmemory_file:
    df = pd.read_parquet(inmemory_file)


prompts=[]
#girl_dictionary=["girl", "woman", "female", "princess",
#            "actless", "maid", "gal", "maiden", "waifu",
#            "wife", "loli", "kawaii"]
girl_dictionary=["girl"]

for idx,row in df.iterrows():
    if(any([x in row["prompt"] for x in girl_dictionary])):
        prompts.append(row["prompt"])

print(len(prompts))

text_model = markovify.Text(prompts)

model_json = text_model.to_json()
with open("models/prompt_gen.json","w") as f:
    f.write(model_json)