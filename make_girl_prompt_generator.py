import markovify
import pandas as pd
import requests
import io

response=requests.get("https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet")
inmemory_file=io.BytesIO(response.content)
with io.BytesIO(response.content) as inmemory_file:
    df = pd.read_parquet(inmemory_file)


prompts=[]
for idx,row in df.iterrows():
    if("girl" in row["prompt"]):
        prompts.append(row["prompt"])

# check max length of words
print(max([max([len(x) for x in prompt.split(" ")]) for prompt in prompts]))

text_model = markovify.Text(prompts)

model_json = text_model.to_json()
with open("models/prompt_gen.json","w") as f:
    f.write(model_json)