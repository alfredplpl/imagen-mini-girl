import os
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler
import torch
import pickle

BATCH_SIZE=8

with open("prompts.pickle","rb") as f:
    prompts=pickle.load(f)

euler_scheduler = EulerDiscreteScheduler.from_config(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=os.environ["HUG_KEY"],
    subfolder="scheduler"
)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=os.environ["HUG_KEY"],
    scheduler=euler_scheduler,
).to("cuda:0")

for i in range(0,len(prompts),BATCH_SIZE):
    images = pipe(prompts[i:i+BATCH_SIZE], num_inference_steps=50,seed=42).images
    for j,image in enumerate(images):
        image.save(os.path.join(os.environ["DS_PATH"], f"{i+j:09d}.png"))
