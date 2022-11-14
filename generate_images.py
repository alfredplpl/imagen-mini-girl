import os
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler
import torch
import pickle

BATCH_SIZE=8
CUDA=0

with open("prompts_1024.pickle", "rb") as f:
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
).to(f"cuda:{CUDA}")

for seed in range(0, 64):
    out_dir=os.path.join(os.environ["DS_PATH"], f"seed_{seed:09d}")
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    for i in range(0,len(prompts),BATCH_SIZE):
        out_path = os.path.join(out_dir, f"{i:09d}.png")
        if(os.path.exists(out_path)):
            continue
        images = pipe(prompts[i:i+BATCH_SIZE], num_inference_steps=50,seed=seed).images
        for j,image in enumerate(images):
            out_path=os.path.join(out_dir, f"{i+j:09d}.png")
            if (not os.path.exists(out_path)):
                image.save(out_path)
