from argparse import ArgumentParser
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import re

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GPTNeoXForCausalLM
from tqdm import tqdm

# hacky import of truthfulqa stuff
import sys
sys.path.append(str(Path(__file__).parent.resolve() / 'TruthfulQA'))
import truthfulqa as tqa


import pandas as pd


def load_llama_model(model_path, cpu_only=False):
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map=None if cpu_only else 'auto')
    #tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_pythia_model(model_name):
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="./pythia-70m/main",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        cache_dir="./pythia-70m/main",
    )
    return model, tokenizer


def build_prompt(question, prompt_style):
    if prompt_style == 'no-examples':
        prompt = 'Q: ' + question + '\n\nA: '
        return prompt

    prompt = ''.join([tqa.presets.preset_map[prompt_style], '\n\nQ: ', question, '\nA: '])
    return prompt


def main(model_name, questions_path, output_path, prompt_style, continue_partial=False):
    with open("config.json") as config_file:
        config = json.load(config_file)

    #layers = [-1, -5, -9, -13]
    layers = [-1, -3]

    if output_path.exists():
        if not continue_partial:
            raise FileExistsError(f'Output dir {output_path} already exists (use --continue-partial to add to results)')
    else: 
        if continue_partial:
            raise FileNotFoundError(f'Passed --continue-partial but {output_path} does not exist')
        output_path.mkdir()
        with (output_path / 'MANIFEST').open('w') as manifest:
            manifest.writelines([
                f'model_name={model_name}',
                f'prompt_style={model_name}',
                f'layers={layers}',
                f'started_at={datetime.utcnow()}',
            ])

    if 'llama' in model_name:
        llama_model_path = Path(config['llama_model_path'])
        model, tokenizer = load_llama_model(llama_model_path)
    elif 'pythia' in model_name:
        model, tokenizer = load_pythia_model(model_name)

    questions = pd.read_csv(questions_path)

    questions_output_cols = [
        'question', 'question_idx', 'next_token_log_prob',
        'next_token_entropy'
    ]
    questions_output_cols += [
        f'activations_layer_{layer_num}' for layer_num in layers
    ]
    answers_output_cols = [
        'question', 'question_idx', 'answer', 'answer_is_correct',
        'all_log_probs', 'avg_log_prob', 'sum_log_prob', 'avg_entropy'
    ]
    answers_output_cols += [
        f'activations_layer_{layer_num}' for layer_num in layers
    ]

    questions_output_path = output_path / 'questions.csv'
    answers_output_path = output_path / 'answers.csv'

    if not questions_output_path.exists():
        # create file with just the the header row
        with questions_output_path.open('w') as csv_file:
            writer = csv.DictWriter(csv_file, questions_output_cols)
            writer.writeheader()
        #questions_output.to_csv(questions_output_path, index=False)
        start_at_idx = 0
    else:
        # get last line in file
        last_line = subprocess.check_output(['tail', '-1', str(questions_output_path)])
        match = re.match('[^,]*,(\d+)', last_line.decode())
        if match is None:
            start_at_idx = 0
        else:
            start_at_idx = int(match.groups()[0]) + 1

    # ensure we're up to the same question for both the question and answer files
    if not answers_output_path.exists():
        if start_at_idx != 0:
            raise RuntimeError('Progress mismatch: already have questions data but no answers file')
        # create file with just the the header row
        with answers_output_path.open('w') as csv_file:
            writer = csv.DictWriter(csv_file, answers_output_cols)
            writer.writeheader()
    else:
        # get last line in file
        last_line = subprocess.check_output(['tail', '-1', str(answers_output_path)])
        match = re.match('[^,]*,(\d+)', last_line.decode())
        if match is None:
            if start_at_idx != 0:
                raise RuntimeError('Progress mismatch: already have questions data but no answers')
        else:
            this_start_at_idx = int(match.groups()[0]) + 1
            if this_start_at_idx != start_at_idx:
                raise RuntimeError(f'Progress mismatch: at {this_start_at_idx} on answers and {start_at_idx} on questions')

    print(f'Starting at question {start_at_idx}')

    model.eval()

    for i, row in questions[start_at_idx:].iterrows():

        question = row['Question']
        print(question)

        # row of results corresponding to this question
        q_results = {}
        q_results['question'] = question
        q_results['question_idx'] = i

        print('Processing question + examples...')
        # do pass of just the question + in-context examples
        prompt = build_prompt(question, prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

        for layer in layers:
            col_name = f'activations_layer_{layer}'
            q_results[col_name] = outputs.hidden_states[layer][0][-1].numpy().tolist()

        next_token_logits = outputs.logits[0][-1]
        log_probs = next_token_logits.log_softmax(0)
        probs = next_token_logits.softmax(0)
        q_results['next_token_entropy'] = - (log_probs * probs).sum().item()
        q_results['next_token_log_prob'] = log_probs.max().item()

        good_answers = tqa.utilities.split_multi_answer(row['Correct Answers'])
        bad_answers = tqa.utilities.split_multi_answer(row['Incorrect Answers'])

        # do passes with each of the possible answers
        a_output_rows = []
        for a_idx, (answer, is_correct) in enumerate(zip(
            good_answers + bad_answers,
            [True] * len(good_answers) + [False] * len(bad_answers)
        )):
            print('Processing answer...')
            # row of results corresponding to this answer for this question
            a_results = {
                'question': question,
                'question_idx': i,
                'answer': answer,
                'answer_is_correct': is_correct,
            }
            full_input = prompt + ' ' + answer
            inputs = tokenizer(full_input, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

            # get probability distribution over tokens at each token position
            log_probs = outputs.logits[0].log_softmax(-1)  # (num_tokens, vocab_size)
            # skip probabilities of question tokens
            # the -1s are because to get the prob of token t we need to look at
            # output at position t-1
            num_prompt_tokens = inputs['input_ids'].size(-1)
            log_probs = log_probs[num_prompt_tokens-1:, :]
            # get only the probs of tokens in the answer
            answer_log_probs = log_probs[range(log_probs.shape[0]), inputs['input_ids'][0]]
            a_results['all_log_probs'] = answer_log_probs.numpy().tolist()
            a_results['avg_log_prob'] = answer_log_probs.mean().item()
            a_results['sum_log_prob'] = answer_log_probs.sum().item()

            # calculate entropy of distribution over tokens 
            probs = outputs.logits[0].softmax(-1)
            probs = probs[num_prompt_tokens-1:, :]
            entropy_per_pos = - (probs * log_probs).sum(-1)
            a_results['avg_entropy'] = entropy_per_pos.mean().item()

            # get activations
            for layer in layers:
                col_name = f'activations_layer_{layer}'
                a_results[col_name] = outputs.hidden_states[layer][0][-1].numpy().tolist()

            a_output_rows.append(a_results)

        print('Writing results')
        # write results
        with questions_output_path.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, questions_output_cols)
            writer.writerow(q_results)
        
        with answers_output_path.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, answers_output_cols)
            writer.writerows(a_output_rows)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', required=True, choices=['pythia-70m', 'llama2-7b-chat'])
    parser.add_argument('-q', '--questions-path', required=True, help='path to questions and answers CSV')
    parser.add_argument('-o', '--output-path', required=True, help='path for creating results directory')
    parser.add_argument('-p', '--prompt-style', choices=['qa'], default='qa',
        help='what to include in prompts before question')
    parser.add_argument('--continue-partial', action='store_true', help='continue a previously started experiment (with the same output path)')

    args = parser.parse_args()
    questions_path = Path(args.questions_path)
    output_path = Path(args.output_path)

    main(args.model, questions_path, output_path, args.prompt_style, args.continue_partial)