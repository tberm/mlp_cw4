"""
Embedding Generation for Language Model for LLAMA and OPT models
Date: 2023-05-26


This script loads sentences from specified CSV files, processes them with a specified LLaMa or OPT model with Hugging Face's transformers library,
and saves the embeddings of the last token of each sentence into new CSV files.

It does the same work as Generate_Embeddings.py but adds (fragile) functionality for LLaMA.

It's based on Amos Azaria's and Tom Mitchell's implementation for their paper `The Internal State of an LLM Knows When it's Lying.' 
https://arxiv.org/abs/2304.13734

It uses the Hugging Face's tokenizer, with the model names specified in a configuration JSON file or by commandline args. 
Model options for OPT include: '6.7b', '2.7b', '1.3b', '350m', 
Model options for LLaMa include: '7B', '13B', '30B', and '65B'. 

The configuration file and/or commandline args also specify whether to remove periods at the end of sentences, which layers of the model to use for generating embeddings,
and the list of datasets to process.

!!!!!!
CAUTION: Because the LLaMa models are not fully publically available, paths for loading those models are hard-coded into the `load_llama_model` function.
!!!!!

If any step fails, the script logs an error message and continues with the next dataset.

Requirements:
- transformers library
- pandas library
- numpy library
- pathlib library
"""

import torch
from transformers import AutoTokenizer, OPTForCausalLM, LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='embedding_extraction.log')

logging.getLogger('').addHandler(logging.StreamHandler())

def load_llama_model(model_path: Path):
    '''
    Initializes and returns a LLaMa model and tokenizer.

    Args:
    model_path: str. Full path to folder containing model weights.

    Returns:
    Tuple[LlamaForCausalLM, LlamaTokenizer]. A tuple containing the loaded LLaMa model and its tokenizer.
    '''
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto")
    #tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def init_model(model_name: str, llama_model_path: Path = None):
    """
    Initializes and returns the model and tokenizer.
    """
    try:
        if model_name in ['7B', '13B', '30B']:
            model, tokenizer = load_llama_model(llama_model_path)
        else:
            model = OPTForCausalLM.from_pretrained("facebook/opt-"+model_name, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+model_name)
    except Exception as e:
        print(f"An error occurred when initializing the model: {str(e)}")
        return None, None
    return model, tokenizer

def load_data(dataset_path: Path, dataset_name: str, true_false: bool = False):
    filename_suffix = "_true_false" if true_false else ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError as e:
        print(f"Dataset file {dataset_file} not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file {dataset_file}: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"No data in CSV file {dataset_file}: {str(e)}")
        return None
    if 'embeddings' not in df.columns:
        df['embeddings'] = pd.Series(dtype='object')
    return df


def process_row(prompt: str, model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a row of data and returns the embeddings.
    """
    if remove_period:
        prompt = prompt.rstrip(". ")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
    embeddings = {}
    for layer in layers_to_use:
        last_hidden_state = outputs.hidden_states[0][layer][0][-1]
        embeddings[layer] = [last_hidden_state.numpy().tolist()]
    return embeddings

#Still not convinced this function works 100% correctly, but it's much faster than process_row.
def process_batch(batch_prompts: List[str], model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Processes a batch of data and returns the embeddings for each statement.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or any other token of your choice

    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

    # Use the attention mask to find the index of the last real token for each sequence
    seq_lengths = inputs.attention_mask.sum(dim=1) - 1  # Subtract 1 to get the index

    batch_embeddings = {}
    for layer in layers_to_use:
        hidden_states = outputs.hidden_states[layer]

        # Gather the hidden state at the last real token for each sequence
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.detach().cpu().numpy().tolist() for embedding in last_hidden_states]

    return batch_embeddings


def save_data(df, output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    """
    Saves the processed data to a CSV file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"embeddings_{dataset_name}{model_name}_{abs(layer)}{filename_suffix}.csv"
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Permission denied when trying to write to {output_file}. Please check your file permissions.")
    except Exception as e:
        print(f"An unexpected error occurred when trying to write to {output_file}: {e}")

def load_results(output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"embeddings_{dataset_name}{model_name}_{abs(layer)}{filename_suffix}.csv"
    return pd.read_csv(output_file)
 

def main():
    """
    Loads configuration parameters, initializes the model and tokenizer, and processes datasets.

    Configuration parameters are loaded from a JSON file named "BenConfigMultiLayer.json". 
    These parameters specify the model to use, whether to remove periods from the end of sentences, 
    which layers of the model to use for generating embeddings, the list of datasets to process, 
    and the paths to the input datasets and output location.

    The script processes each dataset according to the configuration parameters, generates embeddings for 
    each sentence in the dataset using the specified model and layers, and saves the processed data to a CSV file. 
    If processing a dataset or saving the data fails, the script logs an error message and continues with the next dataset.
    """
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
        return
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
        return
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")
        return

    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--model", 
                        help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--layers", nargs='*', 
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*',
                        help="List of dataset names without csv extension. Can leave off 'true_false' suffix if true_false flag is set to True")
    parser.add_argument("--true_false", action="store_true", help="Do you want to append 'true_false' to the dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", action="store_true", help="Include this flag if you want to extract embedding for the last token before the final period.")
    parser.add_argument("--continue_run", action="store_true", help="Continue a half-finished run, adding to an existing results file")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    true_false = args.true_false if args.true_false is not None else config_parameters["true_false"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else config_parameters["batch_size"]
    llama_model_path = Path(config_parameters['llama_model_path']) if 'llama_model_path' in config_parameters else None
    continue_run = args.continue_run
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])


    model_output_per_layer: Dict[int, pd.DataFrame] = {}

    model, tokenizer = init_model(model_name, llama_model_path)
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return
    #I've left this in in case there's an issue with the batch_processing fanciness
    # for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
    #     dataset = load_data(dataset_path, dataset_name, true_false=true_false)
    #     if dataset is None:
    #         continue
    #     for layer in layers_to_process:
    #         model_output_per_layer[layer] = dataset.copy()

    #     for i, row in tqdm(dataset.iterrows(), desc="Row number"):
    #         sentence = row['statement']
    #         embeddings = process_row(sentence, model, tokenizer, layers_to_process, should_remove_period)
    #         for layer in layers_to_process:
    #             model_output_per_layer[layer].at[i, 'embeddings'] = embeddings[layer]
    #         if i % 100 == 0:
    #             logging.info(f"Processing row {i}")

    #     for layer in layers_to_process:
    #         save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period) 

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        # Increase the threshold parameter to a large number
        np.set_printoptions(threshold=np.inf)
        dataset = load_data(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_already_done = len(dataset)
        for layer in layers_to_process:
            if continue_run:
                part_results = load_results(output_path, dataset_name, model_name, layer, should_remove_period)
                model_output_per_layer[layer] = part_results
                num_already_done = min(
                    num_already_done,
                    part_results.index[part_results.embeddings.isnull()][0]
                )
            else:
                model_output_per_layer[layer] = dataset.copy()
                model_output_per_layer[layer]['embeddings'] = pd.Series(dtype='object')
                num_already_done = 0

        num_to_do = len(dataset) - num_already_done
        num_batches = num_to_do // BATCH_SIZE + (num_to_do % BATCH_SIZE != 0)

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            start_idx = batch_num * BATCH_SIZE + num_already_done
            actual_batch_size = min(BATCH_SIZE, len(dataset) - start_idx)
            end_idx = start_idx + actual_batch_size
            batch = dataset.iloc[start_idx:end_idx]
            batch_prompts = batch['statement'].tolist()
            batch_embeddings = process_batch(batch_prompts, model, tokenizer, layers_to_process, should_remove_period)

            for layer in layers_to_process:
                for i, idx in enumerate(range(start_idx, end_idx)):
                    model_output_per_layer[layer].at[idx, 'embeddings'] = batch_embeddings[layer][i]

            if batch_num % 10 == 0:
                logging.info(f"Processing batch {batch_num}")

            for layer in layers_to_process:
                save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period)


if __name__ == "__main__":
    main()
