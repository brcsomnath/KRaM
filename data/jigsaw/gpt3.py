# +
import math
import openai
import pickle
import tiktoken

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


openai.api_key = ''

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

def setup_gpt3(file='openai_api.key'):
  with open(file, 'r') as f:
    openai.api_key = f.readline().strip()
        


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def get_embeddings(texts, model="text-embedding-ada-002"):
  texts = [text.replace("\n", " ") for text in texts]
  return openai.Embedding.create(input = texts, model=model)['data']


def load_jigsaw_raw(PATH='train.csv', gpu='0'):
  df = pd.read_csv(PATH)
  
  label_set = ['buddhist', 
                'christian', 
                'hindu', 
                'jewish', 
                'muslim', 
                'other_religion', 
                'bisexual', 
                'female', 
                'heterosexual', 
                'homosexual_gay_or_lesbian', 
                'male', 
                'other_gender',
                'other_sexual_orientation',
                'transgender']
  
  rows = []
  for index, row in tqdm(df.iterrows()):
      for label in label_set:
          if not math.isnan(row[label]) and row[label] > 0.:
              rows.append(row)
              break
  return rows


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_text_tokens(text, 
                         encoding_name=EMBEDDING_ENCODING, 
                         max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def dump_pkl(content, filename):
  with open(filename, "wb") as file:t
    pickle.dump(content, file)


def chunk_list(lst, chunk_size):
  chunked_list = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
  if len(lst) % chunk_size != 0:
    chunked_list[-1] = lst[-(len(lst) % chunk_size):]
  return chunked_list




def prepare_jigsaw(dataset, protected_label='religion'):
  '''Prepare the full Jigsaw dataset given the protected label.'''

  label_set = []
  if protected_label == 'gender':
    label_set =['bisexual', 
                'female', 
                'heterosexual', 
                'homosexual_gay_or_lesbian', 
                'male', 
                'other_gender', 
                'other_sexual_orientation', 
                'transgender']
  elif protected_label == 'religion':
    label_set = ['buddhist', 
                 'christian', 
                 'hindu', 
                 'jewish', 
                 'muslim', 
                 'other_religion']
  
  sentence_embs = []
  Y = []
  A = []
  
  for row, emb in tqdm(dataset):
    flag = False
    for label in label_set:
      if not math.isnan(row[label]) and row[label] > 0.:
        flag = True
        rows.append(row)
        break
    
    if not flag:
      continue
    
    
    Y.append(int(row['target'] > 0.5))
    
    a = []
    for k in label_set:
      a.append(float(row[k]))
    A.append(a)
    sentence_embs.append(emb)
  
  sentence_embs = np.array(sentence_embs)
  A = np.array(A)
  Y = np.array(Y)
  
  x_train, y_train, z_train, x_test, y_test, z_test = train_test_split(sentence_embs, Y, A, test_size=0.2)
  return x_train, y_train, z_train, x_test, y_test, z_test



if __name__ == '__main__':
  setup_gpt3()
  
  rows = load_jigsaw_raw()
  chunked_rows = chunk_list(rows, chunk_size=8)

  dataset = []
  for row_batch in tqdm(chunked_rows):
    texts = [x['comment_text'] for x in row_batch]
    emb = get_embeddings(texts)
    dataset.extend([(r, e["embedding"]) for r, e in zip(row_batch, emb)])
  
  # caching the data for safety
  dump_pkl(data, "data_openai.pkl")

  for protected_label in ['gender', 'religion']:
    x_train, y_train, z_train, x_test, y_test, z_test = prepare_jigsaw(
      dataset, protected_label=protected_label)

    dump_dict = {
        'x_train': x_train,
        'y_train': y_train,
        'z_train': z_train,
        'x_test': x_test,
        'y_test': y_test,
        'z_test': z_test
    }

    
    dump_pkl(dump_dict, 
             f"jigsaw_{protected_label}_openai.pkl")
  
