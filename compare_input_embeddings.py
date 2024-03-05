"""
Just an experiment to check whether two input strings that start with the same sequence
of tokens get the same internal representations at the shared token positions -- as they
should for an autoregressive model.

Get strange behaviour in the case where sentence B is (sentence A + something else), but
if both sentences have something after their common part, behaviour is as expected.
"""
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch


def compare_hidden_states(model, tokenizer, prompt1, prompt2, position):
    """
    Compare the hidden representations of the two prompts at given token position.
    Returns list of bools corresponding to whether we have equality for each model layer
    """
    inputs1 = tokenizer.encode(prompt1, return_tensors="pt")
    out1 = model.generate(inputs1, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    hs1 = out1.hidden_states[0]
    inputs2 = tokenizer.encode(prompt2, return_tensors="pt")
    out2 = model.generate(inputs2, output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
    hs2 = out2.hidden_states[0]
    print(inputs1)
    print(inputs2)
    return [
        torch.allclose(layer1[:, position, :], layer2[:, position, :])
        for layer1, layer2 in zip(hs1, hs2)
    ]


model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m",
  cache_dir="./pythia-70m/main",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m",
  cache_dir="./pythia-70m/main",
)


print('"Hello world" and "Hello world"')
print('Position 0')
print(compare_hidden_states(model, tokenizer, 'Hello world', 'Hello world', 0))
print('Position 1')
print(compare_hidden_states(model, tokenizer, 'Hello world', 'Hello world', 1))

print('"Hello Bob" and "Hello Alice"')
print('Position 0')
print(compare_hidden_states(model, tokenizer, 'Hello Bob', 'Hello Alice', 0))
print('Position 1')
print(compare_hidden_states(model, tokenizer, 'Hello Bob', 'Hello Alice', 1))

print('"Hello" and "Hello Alice"')
print('Position 0')
print(compare_hidden_states(model, tokenizer, 'Hello', 'Hello Alice', 0))

