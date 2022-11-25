import os
import argparse

from preprocessing import *
from embeddings import *
from dataloader import create_dataloader
from model import GRUModel
from predict import predict
from utils import *


# A dictionary containing the columns and a list of functions to perform on it in order
def preprocess_data(df):
    data_cleaning_pipeline = {
        DATA_COL: [
            to_lower,
            remove_special_words,
            remove_accented_characters,
            remove_html_encodings,
            remove_html_tags,
            remove_url,
            fix_contractions,
            remove_non_alpha_characters,
            remove_extra_spaces,
        ]
    }

    cleaned_data = df.copy()

    # Process all the cleaning instructions
    for col, pipeline in data_cleaning_pipeline.items():
        # Get the column to perform cleaning on
        temp_data = cleaned_data[col].copy()

        # Perform all the cleaning functions sequencially
        for func in pipeline:
            print(f"Starting: {func.__name__}")
            temp_data = func(temp_data)
            print(f"Ended: {func.__name__}")

        # Replace the old column with cleaned one.
        cleaned_data[col] = temp_data.copy()

    return cleaned_data


def embeddings(df):

    _, torch_idx_text, torch_seg_ids = create_tensors_ROBERTA(df.sentences)
    return get_embeddings(torch_idx_text, torch_seg_ids)


def main(file_path, output_path):

    # Load data
    print("\n👉 Reading input file...")
    org_data = read_file(file_path)

    # Preprocess Data
    print("\n👉 Running cleaning and preprocessing pipeline on input...")
    data = preprocess_data(org_data)

    # Create RoBERTa embeddings
    print("\n👉 Creating RoBERTa embeddings for input...")
    data_embeddings = embeddings(data)

    # Create dataloader
    dataloader = create_dataloader(data_embeddings, BATCH_SIZE, NUM_WORKERS)

    # Load model
    print("\n👉 Creating and loading fairness classification model...")
    model = GRUModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(GRU_MODEL_PATH))

    # Do prediction
    print("\n👉 Performing predictions ...")
    preds = predict(model, dataloader)

    # Write unfair statements
    print("\n👉 Writing predictions to output file...")
    write_predictions(output_path, org_data, preds)

    print(f"\n✅ Completed. Find the unfair clauses in {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="TOS Fairness Classification and Obligation Detection",
        description="The program takes in terms of service documents and returns the unfair clauses, and detects obligation.",
        epilog="By CSCI-544 Fall'22 Group 18",
    )

    parser.add_argument("file_path")
    parser.add_argument('-o', '--output_path', required=False, default=OUTPUT_PATH)

    args = parser.parse_args()

    # Change working directory to this file's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main(args.file_path, args.output_path)
