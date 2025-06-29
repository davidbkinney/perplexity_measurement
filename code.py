"""
code.py

Contains functions used to open a .txt file and obtain the perplexity of the text contained within it, according to GPT-2.
"""

# Import packages.
import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd
import re
import torch
from transformers import models
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

#Load the large version of GPT-2.
model_to_use = "gpt2-large"
tokenizer = GPT2TokenizerFast.from_pretrained(model_to_use)
model = GPT2LMHeadModel.from_pretrained(model_to_use,
                                        output_scores=True,
                                        pad_token_id=tokenizer.eos_token_id)

def open_txt(filename:str) -> str:
    """
    Opens and cleans a .txt file containing a classroom transcript.

    Args:
      filename: string of the address for the input file, stored in the
      working directory.

    Returns:
      A string containing the cleaned text contents of the file.
    """
    #Open the file.
    with open(filename,'r', encoding = 'utf-8-sig', errors = 'ignore') as f:
        x = f.readlines()
    #Tokenize the text contents.
    for i in range(len(x)):
        x[i] = nltk.tokenize.word_tokenize(x[i])

    #Clean the text contents.
    x = list(map(str, x))
    x = [re.sub(r'\w*:\w*', '', word).strip() for word in x if ':' in word] #remove colons
    x = [re.sub(r'\\\\n|\\\\t', '', word) for word in x] # remove line breaks and tab breaks
    x = [re.sub(r'[^\w\s]|_', '', word) for word in x] # remove punctuation and underscore
    x = [re.sub(r'\d{1, 3}', '', word) for word in x] # remove digits that are a minimum of 1 and a maximum of 3
    x = [re.sub(r'\w*\d\w*', '', word) for word in x] # remove character strings that contain a digit
    x = [re.sub(r'teacher', '', word) for word in x] # remove character strings that the word 'teacher'
    x = [word.lower() for word in x]
    x = [word.split() for word in x]
    for i in range(len(x)):
        filt = []
        for j in range(len(x[i])):
            if x[i][j] != 'teacher' and x[i][j] != 'female' and x[i][j] != 'male':
                filt.append(x[i][j])
        x[i] = filt
    return ' '.join([' '.join(doc) for doc in x])


def get_perplexity(text:str) -> tensor:
  """Obtains the perplexity of the text in a string."""
  encodings = tokenizer(text, return_tensors="pt")
  max_length = model.config.n_positions
  stride = 512
  seq_len = encodings.input_ids.size(1)
  nlls = []
  prev_end_loc = 0
  for begin_loc in range(0, seq_len, stride):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc]
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)
          neg_log_likelihood = outputs.loss

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break

  ppl = torch.exp(torch.stack(nlls).mean())
  return ppl
