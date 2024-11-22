import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens import Doc
import numpy as np
import json
from src.nx.nx_utils import spacy2pdf

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
json_path = '/home/pgajo/Multitask-RFG/data/yamakata/efrc_ud.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

sample = data[6]

# Load the English language model
nlp = spacy.load('en_core_web_sm')

sent = ' '.join(sample['words'])
doc = nlp(Doc(nlp.vocab, sent))
# doc = nlp( sent)

spacy2pdf(doc, filename = './prints/dep_tree.pdf')

# Get the number of tokens
num_tokens = len(doc)

# Initialize an adjacency matrix of size num_tokens x num_tokens with zeros
A = np.zeros((num_tokens, num_tokens), dtype=int)

# Populate the adjacency matrix based on dependencies with only `dobj` labels
for token in doc:
    if token.head != token:  # Only add for `dobj` dependencies
        A[token.head.i, token.i] = 1  # Edge from head to token for `dobj`

print("sent")
print(sent)

print("adjaceny matrix A")
print(A)
print(A.shape)

print("step_indices")
step_indices = np.array(sample['step_indices'])
print(step_indices)

def make_step_intermediate_matrix(step_indices):
    L = len(step_indices)         # Number of words
    K = np.max(step_indices) + 1  # Number of unique steps

    # Initialize P as an L x K matrix of zeros
    M = np.zeros((L, K), dtype=int)

    # Fill M based on step_indices
    for i, step in enumerate(step_indices):
        M[i, step] = 1
    return M

P = make_step_intermediate_matrix(step_indices)

intermediate = A @ P
print(intermediate, intermediate.shape)

B = P.T @ A @ P
print(B)