# See Also: https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/t5.py
import os.path
import torch

from transformers import T5EncoderModel,T5Tokenizer
import pickle
from einops import rearrange

MAX_LENGTH = 32
VECTOR_SIZE=768
BATCH_SIZE=128

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

with open("prompts_1024.pickle", "rb") as f:
    prompts=pickle.load(f)

t5=T5EncoderModel.from_pretrained(DEFAULT_T5_NAME)
tokenizer = T5Tokenizer.from_pretrained(DEFAULT_T5_NAME, model_max_length=MAX_LENGTH)
t5.cuda()
t5.eval()

text_embeds = torch.zeros((len(prompts), MAX_LENGTH, VECTOR_SIZE), device="cpu")
text_masks = torch.zeros((len(prompts), MAX_LENGTH), device="cpu", dtype=torch.bool)

for i in range(0,len(prompts),BATCH_SIZE):
    with torch.no_grad():
        encoded = tokenizer.batch_encode_plus(
            prompts[i:i+BATCH_SIZE],
            return_tensors="pt",
            padding='max_length',
            max_length=MAX_LENGTH,
            truncation=True
        )
        token_ids = encoded.input_ids.cuda()
        attn_mask = encoded.attention_mask.cuda()

        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    attn_mask = attn_mask.bool()

    # just force all embeddings that is padding to be equal to 0.
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.)

    text_embeds[i:i+BATCH_SIZE]=encoded_text
    text_masks[i:i+BATCH_SIZE]=attn_mask

torch.save({"text_embeds":text_embeds,"text_masks":text_masks},
           os.path.join(os.environ["DS_PATH"], "text_embedding_1024.pickle"))
