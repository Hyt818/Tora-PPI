'''
Purpose: Extract protein language model representation based on protein sequence, used to generate sequence features required for subsequent training.
Input: Protein sequence dictionary file, pre-trained sequence model/tokenizer path or name.
Output: Protein sequence-level or residue-level embedding files, for loading by the training script. 
'''

import os
import re
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_protein(data_path):
    protein = []
    protein_id = []

    for line in tqdm(open(data_path), desc="Reading protein sequences"):
        line = line.strip().split('\t')
        if len(line) < 2:
            continue
        if line[0] not in protein_id:
            protein_id.append(line[0])
            protein.append((line[0], line[1]))

    lengths = [len(p[1]) for p in protein]
    print("protein num:", len(protein))
    print("protein average length:", np.average(lengths))
    print("protein max & min length:", np.max(lengths), np.min(lengths))
    print('-----------------------------------------------------------------------')
    return protein

def preprocess_seqs(seq_list):
    processed = []
    for s in seq_list:
        s = re.sub(r"[UZOB]", "X", s)
        s = " ".join(list(s))
        processed.append(s)
    return processed


def main():
    LOCAL_MODEL_DIR = "./prottrans"
 
    os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./hf_cache")

    tokenizer = T5Tokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        do_lower_case=False,
        local_files_only=True
    )
    model = T5EncoderModel.from_pretrained(
        LOCAL_MODEL_DIR,
        local_files_only=True
    ).to(device)
    model.eval()

    data_path = './data/SHS27K/protein.SHS27k.sequences.dictionary.tsv'
    proteins = get_protein(data_path)

    batch_size = 1
    num_batches = math.ceil(len(proteins) / batch_size)

    acid_representations = []
    sequence_representations = []

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_data = proteins[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        seqs = [p[1] for p in batch_data]
        seqs_processed = preprocess_seqs(seqs)

        ids = tokenizer(
            seqs_processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors=None
        )

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_state = outputs.last_hidden_state  # [B, L, 1024]

        for i in range(input_ids.size(0)):
            real_len = int(attention_mask[i].sum().item())
            residue_emb = last_state[i, 1:real_len - 1]
            acid_representations.append(residue_emb.cpu().numpy())

            protein_emb = residue_emb.mean(dim=0)
            sequence_representations.append(protein_emb.cpu().numpy())

    torch.save(acid_representations, 'ProtT5_residue_Embedding.pt')
    torch.save(sequence_representations, 'ProtT5_sequence_Embedding.pt')
    print("Saved:", 'ProtT5_residue_Embedding.pt', 'ProtT5_sequence_Embedding.pt')


if __name__ == "__main__":
    main()
