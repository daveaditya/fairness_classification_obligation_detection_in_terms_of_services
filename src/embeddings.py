import logging
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

from utils import device

cls = "[CLS]"
sep = "[SEP]"
pad = "[PAD]"
bert_pad_len = 512


def create_tensors_ROBERTA(text):
  """
    Tokenize using ROBERTA Tokenizer for the pd.Series
  """
  print("Tokenizing text...")
  logging.basicConfig(level = logging.INFO)

  # Load the `bert-base-uncased` model
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

  # Tokenize every sentence in the pd.Series
  tokenized_text = [tokenizer.tokenize(x) for x in text]

  # Pad the tokens to be used for BERT Model; BERT takes fixed lengend sequence
  tokenized_text = [x + ([pad] * (bert_pad_len - len(x))) for x in tokenized_text]

  # Convert the tokens to their IDs
  indexed_text = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

  # BERTModel has Q&A format, so setting the context to one for every sentence
  segment_ids = [[1] * len(x) for x in tokenized_text]

  # Convert to tensor
  torch_idx_text = torch.LongTensor(indexed_text)
  torch_seg_ids = torch.LongTensor(segment_ids)
  
  return tokenized_text, torch_idx_text, torch_seg_ids     


#takes in the index and segment tensors and returns the bert embeddings as a list
def get_embeddings(torch_idx_text, torch_seg_ids):
    """
      Create RoBERTa embeddings from tokens
    """
    print("Getting Embeddings...")

    # Load pretrained `roberta-base` model, and set to inference
    model = RobertaModel.from_pretrained('roberta-base', output_hidden_states = True)
    model.eval()

    torch_idx_text, torch_seg_ids = torch_idx_text.to("cpu"), torch_seg_ids.to("cpu")
    model.to(device)

    # Disable gradient and get BERT embeddings
    with torch.no_grad():
        roberta_embeddings = []
        for i in range(len(torch_idx_text)):
            print(i, end = "\r")
            text_temp = torch.unsqueeze(torch_idx_text[i], dim = 0).to(device)
            sgmt_temp = torch.unsqueeze(torch_seg_ids[i], dim = 0).to(device)
            output = model(text_temp, sgmt_temp)
            roberta_embeddings.append(output[0])
            del text_temp, sgmt_temp
    del model
    return roberta_embeddings


def embeddings_to_words(tokenized_text, embeddings):
    """
      Clubbing same word tokens to reduce dimensionality
      Note: Need to run this locally and tweak virtual memory, as the colab
      runtime crashes
    """
    print("Untokenizing text and embeddings...")
    embeddings = [x.cpu().detach().numpy() for x in embeddings]
    embeddings = np.concatenate(embeddings, axis = 0)
    sentences = []
    final_emb = []

    # Iterate over every sentence
    for i in range(len(tokenized_text)):
        txt = tokenized_text[i]
        sub_len = 0
        sent = []
        sub = []
        emb = []
        sub_emb = None
        try:
            idx = txt.index(pad)
        except:
            idx = len(txt)
        for j in range(idx):
            # For the token that starts with ## process it; remove ## and
            # club that with previous token;
            # For the embedding, take the average of token's embeddings
            if txt[j].startswith("##"):
                if sub == []:
                    sub.append(sent.pop())
                    emb.pop()
                    sub_emb = embeddings[i][j - 1] + embeddings[i][j]
                    sub.append(txt[j][2:])
                    sub_len = 2
                else:
                    sub.append(txt[j][2:])
                    sub_emb += embeddings[i][j]
                    sub_len += 1
            else:
                if sub != []:
                    sent.append("".join(sub))
                    emb.append(sub_emb / sub_len)
                    sub = []
                    sub_emb = None
                    sub_len = 0
                sent.append(txt[j])
                emb.append(embeddings[i][j])
        sentences.append(sent)
        final_emb.append(emb)
    return sentences, final_emb