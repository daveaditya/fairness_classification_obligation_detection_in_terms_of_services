import pandas as pd
import torch

GRU_MODEL_PATH = "./../model/gru_bert_model.pt"
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
OUTPUT_DIM = 2

# how many samples per batch to load
BATCH_SIZE = 20

# number of subprocesses to use for data loading
NUM_WORKERS = 0

DATA_COL = "sentences"

OUTPUT_PATH = "./../results/output.txt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Some features are not available for MPS
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


def read_file(file_path):
    return pd.read_csv(file_path, sep="\n", names=[DATA_COL])


def write_predictions(file_path, df, preds, only_unfair=True):
    with open(file_path, mode="w") as file:
        for i in range(len(df)):
            if only_unfair:
                if preds[i]:
                    file.write(f"{df.loc[i, DATA_COL]}\n")
            else:
                file.write(f"{df.loc[i, DATA_COL]}, {preds[i]}\n")

device = get_device()
