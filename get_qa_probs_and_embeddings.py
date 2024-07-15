"""
Uses a LLM (LLama or Pythia) to process question-answer pairs and save the model's
embeddings and token probabilities.  We save the embeddings at the final token position
of the answer over several layers (hardcoded in run_gen/run_mc) and aggregate over token
probabilities for all the tokens in the answer. Results are written to a CSV file.
"""
from argparse import ArgumentParser
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import re

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, GPTNeoXForCausalLM
from tqdm import tqdm

# hacky import of truthfulqa stuff
import sys
sys.path.append(str(Path(__file__).parent.resolve() / 'TruthfulQA'))
import truthfulqa as tqa


import pandas as pd


def load_llama_model(model_path, cpu_only=False):
    model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, device_map=None if cpu_only else 'auto')
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


def print_tokens_with_probs(tokens, tokenizer, log_probs):
    lines = [
        f'{tokenizer.decode(token)} ({log_prob})'
        for token, log_prob in zip(tokens, log_probs)
    ]
    print('\n'.join(lines))


def run_gen(model_name, questions_path, output_path, prompt_style, continue_partial=False, cpu_only=False, debug=False):
    """
    Get embeddings and token probabilities for model-generated answers.
    ``questions_path`` should point to a CSV containing a column for questions and a
    column for model-generated answers.
    """
    with open("config.json") as config_file:
        config = json.load(config_file)

    layers = [-1, -5, -9, -13]

    if output_path.exists():
        if not continue_partial:
            raise FileExistsError(f'Output dir {output_path} already exists (use --continue-partial to add to results)')
    else: 
        if continue_partial:
            raise FileNotFoundError(f'Passed --continue-partial but {output_path} does not exist')
        output_path.mkdir()
        with (output_path / 'MANIFEST').open('w') as manifest:
            manifest.writelines([
                f'model_name={model_name}\n',
                f'prompt_style={prompt_style}\n',
                f'layers={layers}\n',
                f'started_at={datetime.utcnow()}\n',
            ])

    if 'llama' in model_name:
        llama_model_path = Path(config['llama_model_path'])
        model, tokenizer = load_llama_model(llama_model_path, cpu_only)
    elif 'pythia' in model_name:
        model, tokenizer = load_pythia_model(model_name)

    device = 'cpu' if cpu_only else 'cuda:0'
 
    questions = pd.read_csv(questions_path)

    output_cols = [
        'question', 'question_idx', 'answer', 'answer_is_correct', 'avg_log_prob', 'sum_log_prob',
    ] + [
        f'activations_layer_{layer_num}' for layer_num in layers
    ]

    output_path = output_path / 'answers.csv'

    if not output_path.exists():
        # create file with just the the header row
        with output_path.open('w') as csv_file:
            writer = csv.DictWriter(csv_file, output_cols)
            writer.writeheader()
        start_at_idx = 0
    else:
        # get last line in file
        last_line = subprocess.check_output(['tail', '-1', str(output_path)])
        match = re.match(r'[^,]*,(\d+)', last_line.decode())
        if match is None:
            start_at_idx = 0
        else:
            start_at_idx = int(match.groups()[0]) + 1

    print(f'Starting at question {start_at_idx}')

    model.eval()

    pbar = tqdm(questions[start_at_idx:].iterrows(), total=len(questions) - start_at_idx)
    for i, row in pbar:
        question = row['Question']
        answer = row[model_name]

        pbar.set_description(question)

        # row of results corresponding to this question
        results = {}
        results['question'] = question
        results['question_idx'] = i
        results['answer'] = row[model_name]

        prompt = build_prompt(question, prompt_style)
        # tokenise just so we know how long the prompt is
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        num_prompt_tokens = prompt_tokens['input_ids'].size(-1)
        full_input = prompt + answer
        inputs = tokenizer(full_input, return_tensors="pt").to(device)
        # last token of prompt input gets replaced by first token of answer in full
        # tokenised sequence. So answer starts at index num_prompt_tokens-1 and question
        # ends at the token before that. 
        assert inputs['input_ids'][0][num_prompt_tokens-2] == prompt_tokens['input_ids'][0][-2]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

        # get probability distribution over tokens at each token position
        log_probs = outputs.logits[0].log_softmax(-1)  # (num_tokens, vocab_size)
        # skip probabilities of question tokens
        # the -1s are because to get the prob of token t we need to look at
        # output at position t-1
        log_probs = log_probs[num_prompt_tokens-2:-1, :]
        # get only the probs of tokens in the answer
        answer_tokens = inputs['input_ids'][0][num_prompt_tokens-1:]
        answer_log_probs = log_probs[range(log_probs.shape[0]), answer_tokens]
        results['avg_log_prob'] = answer_log_probs.mean().cpu().item()
        results['sum_log_prob'] = answer_log_probs.sum().cpu().item()

        if debug:
            print_tokens_with_probs(answer_tokens, tokenizer, answer_log_probs)
            print('Average log prob:', answer_log_probs.mean().cpu().item())
 
        # get activations
        for layer in layers:
            col_name = f'activations_layer_{layer}'
            results[col_name] = outputs.hidden_states[layer][0][-1].cpu().numpy().tolist()

        # write results
        with output_path.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, output_cols)
            writer.writerow(results)


def run_mc(model_name, questions_path, output_path, prompt_style, continue_partial=False, cpu_only=False, debug=False):
    """
    Get embeddings and token probabilities for question-answer pairs given in the
    TruthfulQA multiple choice setting. Since there are several answers for each
    question, the output is two separate CSVs: one for questions (recording token
    probabilities and entropy after the question) and one for answers (recording
    embeddings and token probabilities of the answer).
    
    ``questions_path`` should point to the CSV of questions and answers from TruthfulQA.
    """
    with open("config.json") as config_file:
        config = json.load(config_file)

    layers = [-1, -5, -9, -13]

    if output_path.exists():
        if not continue_partial:
            raise FileExistsError(f'Output dir {output_path} already exists (use --continue-partial to add to results)')
    else: 
        if continue_partial:
            raise FileNotFoundError(f'Passed --continue-partial but {output_path} does not exist')
        output_path.mkdir()
        with (output_path / 'MANIFEST').open('w') as manifest:
            manifest.writelines([
                f'model_name={model_name}\n',
                f'prompt_style={prompt_style}\n',
                f'layers={layers}\n',
                f'started_at={datetime.utcnow()}\n',
            ])

    if 'llama' in model_name:
        llama_model_path = Path(config['llama_model_path'])
        model, tokenizer = load_llama_model(llama_model_path, cpu_only)
    elif 'pythia' in model_name:
        model, tokenizer = load_pythia_model(model_name)

    device = 'cpu' if cpu_only else 'cuda:0'
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
        match = re.match(r'[^,]*,(\d+)', last_line.decode())
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
        match = re.match(r'[^,]*,(\d+)', last_line.decode())
        if match is None:
            if start_at_idx != 0:
                raise RuntimeError('Progress mismatch: already have questions data but no answers')
        else:
            this_start_at_idx = int(match.groups()[0]) + 1
            if this_start_at_idx != start_at_idx:
                raise RuntimeError(f'Progress mismatch: at {this_start_at_idx} on answers and {start_at_idx} on questions')

    print(f'Starting at question {start_at_idx}')

    model.eval()

    pbar = tqdm(questions[start_at_idx:].iterrows(), total=len(questions) - start_at_idx)
    for i, row in pbar:
        question = row['Question']

        pbar.set_description(question)

        # row of results corresponding to this question
        q_results = {}
        q_results['question'] = question
        q_results['question_idx'] = i

        # do pass of just the question + in-context examples
        prompt = build_prompt(question, prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_prompt_tokens = inputs['input_ids'].size(-1)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

        for layer in layers:
            col_name = f'activations_layer_{layer}'
            q_results[col_name] = outputs.hidden_states[layer][0][-1].cpu().numpy().tolist()

        next_token_logits = outputs.logits[0][-1]
        log_probs = next_token_logits.log_softmax(0)
        probs = next_token_logits.softmax(0)
        q_results['next_token_entropy'] = - (log_probs * probs).sum().cpu().item()
        q_results['next_token_log_prob'] = log_probs.max().cpu().item()

        good_answers = tqa.utilities.split_multi_answer(row['Correct Answers'])
        bad_answers = tqa.utilities.split_multi_answer(row['Incorrect Answers'])

        # do passes with each of the possible answers
        a_output_rows = []
        for a_idx, (answer, is_correct) in enumerate(zip(
            good_answers + bad_answers,
            [True] * len(good_answers) + [False] * len(bad_answers)
        )):
            # row of results corresponding to this answer for this question
            a_results = {
                'question': question,
                'question_idx': i,
                'answer': answer,
                'answer_is_correct': is_correct,
            }
            full_input = prompt + answer
            inputs = tokenizer(full_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True) 

            # get probability distribution over tokens at each token position
            log_probs = outputs.logits[0].log_softmax(-1)  # (num_tokens, vocab_size)
            # skip probabilities of question tokens
            # the -1s are because to get the prob of token t we need to look at
            # output at position t-1
            log_probs = log_probs[num_prompt_tokens-2:-1:, :]
            # get only the probs of tokens in the answer
            answer_tokens = inputs['input_ids'][0][num_prompt_tokens-1:]
            answer_log_probs = log_probs[range(log_probs.shape[0]), answer_tokens]
            a_results['all_log_probs'] = answer_log_probs.cpu().numpy().tolist()
            a_results['avg_log_prob'] = answer_log_probs.mean().cpu().item()
            a_results['sum_log_prob'] = answer_log_probs.sum().cpu().item()


            if debug:
                print_tokens_with_probs(answer_tokens, tokenizer, answer_log_probs)
                print('Average log prob:', answer_log_probs.mean().cpu().item())
            
            # calculate entropy of distribution over tokens 
            probs = outputs.logits[0].softmax(-1)
            probs = probs[num_prompt_tokens-2:-1, :]
            entropy_per_pos = - (probs * log_probs).sum(-1)
            a_results['avg_entropy'] = entropy_per_pos.mean().cpu().item()

            # get activations
            for layer in layers:
                col_name = f'activations_layer_{layer}'
                a_results[col_name] = outputs.hidden_states[layer][0][-1].cpu().numpy().tolist()

            a_output_rows.append(a_results)

        # write results
        with questions_output_path.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, questions_output_cols)
            writer.writerow(q_results)
        
        with answers_output_path.open('a') as csv_file:
            writer = csv.DictWriter(csv_file, answers_output_cols)
            writer.writerows(a_output_rows)

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', required=True, choices=['pythia-70m', 'llama2-7b-chat'])
    parser.add_argument('-q', '--questions-path', required=True, help='path to questions and answers CSV')
    parser.add_argument('-o', '--output-path', required=True, help='path for creating results directory')
    parser.add_argument('-p', '--prompt-style', choices=['qa', 'help', 'harm', 'null'], default='qa',
        help='which of the TruthfulQA prompt formats to use')
    parser.add_argument('--continue-partial', action='store_true', help='continue a previously started experiment (with the same output path)')
    parser.add_argument('--cpu-only', action='store_true', help='do not use GPU')
    parser.add_argument('--qa-mode', choices=['mc', 'gen'],
        help='run on model-generated answers or on the answers provided in the TruthfulQA multiple choice task (mc)',
        required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    questions_path = Path(args.questions_path)
    output_path = Path(args.output_path)

    if args.qa_mode == 'mc':
        run_mc(args.model, questions_path, output_path, args.prompt_style, args.continue_partial, args.cpu_only)
    elif args.qa_mode == 'gen':
        run_gen(args.model, questions_path, output_path, args.prompt_style, args.continue_partial, args.cpu_only, args.debug)
