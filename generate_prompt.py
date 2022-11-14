import os
import pickle
import time

import markovify
from concurrent.futures  import ProcessPoolExecutor
from tqdm import tqdm

from transformers import CLIPTokenizer

LIMIT_CLIP_TEXT_ENC=75

girl_dictionary = ["girl", "woman", "female", "princess",
              "actless", "maid", "gal", "maiden", "waifu",
              "wife", "loli", "kawaii"]
boy_dictionary = ["boy", "man", "male", "guy",
                  "dude", "geezer", "chap", "fellow","bloke"]
def generate_girl_prompt(model,tokenizer):
    prompt=model.make_sentence()

    if (prompt is None):
        return None

    tokens = tokenizer.tokenize(prompt)

    if (len(tokens) > LIMIT_CLIP_TEXT_ENC):
        return None

    if (any([x in prompt for x in girl_dictionary])):
        if (all([not x in prompt for x in boy_dictionary])):
            return prompt

    return None

def compile(model_json):
    model = markovify.Text.from_json(model_json)
    model = model.compile()
    return model

if __name__ == "__main__":
    MAX_WORKERS=os.cpu_count()
    NUM_PROMPTS=2**12+2**11
    prompts=[]

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with open("models/prompt_gen.json", "r") as f:
        model_json = f.read()

    model=compile(model_json)

    with tqdm(total=NUM_PROMPTS) as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(MAX_WORKERS):
                futures.append(executor.submit(generate_girl_prompt, model, tokenizer))

            while(True):
                for i in range(MAX_WORKERS):
                    if(futures[i].done()):
                        prompt=futures[i].result()
                        if(prompt is not None):
                            print(prompt)
                            prompts.append(prompt)
                            pbar.update(1)
                        futures[i]=executor.submit(generate_girl_prompt, model, tokenizer)
                if (len(prompts) > NUM_PROMPTS):
                    break

    prompts=prompts[:NUM_PROMPTS]
    with open(f"prompts_{NUM_PROMPTS}.pickle", "wb") as f:
        pickle.dump(prompts,f,pickle.HIGHEST_PROTOCOL)
