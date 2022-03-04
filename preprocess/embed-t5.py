import tqdm
import dataset
from dataset import *


# %% [markdown]
# ## Important Notes:
# 1. ProtT5-XL-UniRef50 has both encoder and decoder, for feature extraction we only load and use the encoder part.
# 2. Loading only the encoder part, reduces the inference speed and the GPU memory requirements by half.
# 2. In order to use ProtT5-XL-UniRef50 encoder, you must install the latest huggingface transformers version from their GitHub repo.
# 3. If you are intersted in both the encoder and decoder, you should use T5Model rather than T5EncoderModel.

# %% [markdown]
# <h3>Extracting protein sequences' features using ProtT5-XL-UniRef50 pretrained-model</h3>

# %% [markdown]
# **1. Load necessry libraries including huggingface transformers**

# %%
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
import argparse

# %% [markdown]
# <b>2. Load the vocabulary and ProtT5-XL-UniRef50 Model<b>


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--split', type=str)
# name = "prot_t5_xxl_uniref50"
args = parser.parse_args()
# name = "prot_t5_xxl_uniref50"
# print(name)
name = args.model

split = args.split
fluorescence = LMDBDataset(f'./data/fluorescence/fluorescence_{split}.lmdb')
print(len(fluorescence))

# %%
tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{name}", do_lower_case=False)

# %%
model = T5EncoderModel.from_pretrained(f"Rostlab/{name}")

# %%
gc.collect()

# %% [markdown]
# <b>3. Load the model into the GPU if avilabile and switch to inference mode<b>

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = model.to(device)
model = model.eval()

# %% [markdown]
# <b>4. Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)<b>

# %%
data = [' '.join(i['primary']) for i in fluorescence]
# print(fluorescence[0])
# print(fluorescence[1])
ret = [i for i in fluorescence]


def process_samples(sequences_Example):
    # sequences_Example = ["A E T C Z A O","S K T Z P"]
    # sequences_Example = data

    # %%
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence)
                         for sequence in sequences_Example]

    # %% [markdown]
    # <b>5. Tokenize, encode sequences and load it into the GPU if possibile<b>

    # %%
    ids = tokenizer.batch_encode_plus(
        sequences_Example, add_special_tokens=True, padding=True)

    # %%
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # %% [markdown]
    # <b>6. Extracting sequences' features and load it into the CPU if needed<b>

    # %%
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # %%
    embedding = embedding.last_hidden_state.cpu().numpy()

    # %% [markdown]
    # <b>7. Remove padding (\<pad\>) and special tokens (\</s\>) that is added by ProtT5-XL-UniRef50 model<b>

    # %%
    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)
    return features


batch_size = 32


def ceil_div(a, b):
    return - (a // -b)


for i in tqdm.tqdm(range(ceil_div(len(data), batch_size))):
    embed = process_samples(data[i * batch_size: (i + 1) * batch_size])
    print([e.shape for e in embed])
    for idx in range(batch_size):
        try:
            ret[idx + i * batch_size]['embed'] = embed[idx]
        except:
            print(idx + i * batch_size)
            continue
    # print(len(tot))

torch.save(ret, f'{name}_{split}.pt')
