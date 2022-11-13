# See Also: https://github.com/lucidrains/imagen-pytorch

import torch
import os

from imagen_pytorch import ElucidatedImagenConfig, ImagenTrainer
from torchvision import transforms
from imagen_pytorch.data import Dataset
from PIL import Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOOP_MAX = 1024 ** 3
STORE_DIR = os.environ["STORE_DIR"]
EMBED_PATH = os.path.join(os.environ["DS_PATH"], "text_embedding_1024.pickle")

SIZE = 64
MAX_BATCH_SIZE = 32
INTERVAL = 400*4

imagen = ElucidatedImagenConfig(
    unets = [
        dict(dim = 128,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False)
    ],
    image_sizes=(64,),
    sigma_min=0.002,  # min noise level
    sigma_max=80,  # max noise level
    sigma_data=0.5,  # standard deviation of data distribution
    rho=7,  # controls the sampling schedule
    P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
    P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn=40,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin=0.005,
    S_tmax=50,
    S_noise=1.003,
).create()

trainer = ImagenTrainer(imagen)

class CustomDataset(Dataset):
    def __init__(
        self,
        image_size,
        embed_path
    ):
        super().__init__(
            "",
            image_size
        )
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
            ])

        pth = torch.load(embed_path)
        self.text_embeds = pth["text_embeds"]
        self.text_masks = pth["text_masks"]

    def __len__(self):
        return self.text_embeds.size()[0]

    def __getitem__(self, index):
        path = os.path.join(os.environ["DS_PATH"],f"{index:09d}.png")
        img = Image.open(path)
        img=self.transform(img)
        text_embed = self.text_embeds[index]
        text_mask = self.text_masks[index]

        return img,text_embed,text_mask

dataset = CustomDataset(SIZE, EMBED_PATH)
trainer.add_train_dataset(dataset, batch_size=64, num_workers=6, shuffle=True)

with tqdm(initial=0, total=LOOP_MAX) as pbar:
    for i in range(0, LOOP_MAX):
        loss = trainer.train_step(unet_number=1, max_batch_size=MAX_BATCH_SIZE)
        pbar.set_description(f'loss: {loss:.5f}')

        if (i % INTERVAL == 0 and i != 0):  # is_main makes sure this can run in distributed
            filename = f'model-{i}.pt'
            trainer.save(os.path.join(STORE_DIR, filename))
        pbar.update(1)
