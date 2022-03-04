from preprocess.dataset import *
import torch
import os
import subprocess


def generate_fasta(split):
    fluorescence = LMDBDataset(
        f'/root/yijia/ProtTrans/data/fluorescence/fluorescence_{split}.lmdb')
    with open(f"/root/yijia/esm/data/fluorescence_{split}.fasta", 'w') as f:
        for i in fluorescence:
            f.write(f">{str(i['id']).zfill(5)}_{i['log_fluorescence'][0]}\n")
            f.write(i['primary'] + '\n')


def embed(split):
    cmd = f"""python extract.py esm1b_t33_650M_UR50S data/fluorescence_{split}.fasta data/embed/esm1b_{split}/ \
        --include mean per_tok bos"""
    # os.system(cmd)
    subprocess.run(cmd.split())


def merge(split):
    fluorescence = LMDBDataset(
        f'/root/yijia/ProtTrans/data/fluorescence/fluorescence_{split}.lmdb')
    ret = [i for i in fluorescence]

    for i in range(len(ret)):
        os.makedirs(f"./data/embed/esm1b_{split}", exist_ok=True)
        name = f"{str(ret[i]['id']).zfill(5)}_{ret[i]['log_fluorescence'][0]}"
        path = f"./data/embed/esm1b_{split}/{name}.pt"
        embed = torch.load(path)
        ret[i]['embed'] = embed

    torch.save(ret, f'esm1b_{split}.pt')


for s in ['train', 'valid', 'test']:
    # for s in ['valid', 'test']:
    generate_fasta(s)
    embed(s)
    merge(s)
