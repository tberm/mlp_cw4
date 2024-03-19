from itertools import islice
import pandas as pd
import sys

'''


python3 fix_gpt_evals.py llama-qa-results-gen-prof-prompt/generated_answers_and_evals.csv 0


'''

def print_bold(*text):
    print(f'\033[1m{" ".join(text)}\033[0m')


if __name__ == '__main__':
    file = sys.argv[1]
    try:
        start_from = int(sys.argv[2])
    except IndexError:
        start_from = 0

    print(f'Reading {file}')
    print(f'Starting from question {start_from}')

    df = pd.read_csv(file)
    for i, row in islice(df.iterrows(), start_from, None):
        if 'I have no comment' in row['answer']:
            print(f'Skipping question {i} as LLAMA has no comment.')
            df.loc[i, 'GPT eval'] = 'true'
            continue
        if row['GPT eval'] == 'true':
            gpt_says_correct = True
        elif row['GPT eval'] == 'false':
            gpt_says_correct = False
        else:
            print(f'No GPT eval for question {i}!')
            continue
        print(f"Question index number {i}: {row['Question']}")
        print()
        print_bold('LLAMA ANSWER:', row['answer'])
        print()
        if gpt_says_correct:
            print_bold('GOOD ANSWER:', row['Best Answer'])
            print('BAD ANSWER:', row['Incorrect Answers'].split(';')[0])
        else:
            print('GOOD ANSWER:', row['Best Answer'])
            print_bold('BAD ANSWER:', row['Incorrect Answers'].split(';')[0])
        while True:
            resp = input(f'GPT says this is {gpt_says_correct}. Agree? [enter if yes, type "no" if not]')
            if resp.strip() == '':
                break
            if resp.strip() == 'no':
                print(f'Changing eval to {not gpt_says_correct}')
                df.loc[i, 'GPT eval'] = 'false' if gpt_says_correct else 'true'
                df.to_csv(file)
                break

        print()
        print('---------------------------------------------------------------------------------')
        print()
