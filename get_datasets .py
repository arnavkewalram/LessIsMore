import datasets
import torch
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json
import os
import csv

ANSWER_PROMPT = "The final answer is: "

def _split_name_fraction(name: str) -> Tuple[str, Optional[float]]:
    """
    Split a dataset name that may include a trailing ':fraction' (e.g., 'boolq:0.2').
    Returns (base_name, fraction or None if absent).
    """
    if ":" in name:
        base, frac = name.split(":", 1)
        try:
            return base, float(frac)
        except ValueError:
            return base, None
    return name, None

def get_default(name: str, split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 1, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    dataset = []
    with open(f"tasks/{name}/{split}.json") as f:
        dataset = json.loads(f.read())
    num_conversations = len(dataset)
    dataset = dataset[:int(num_conversations * data_fraction)]
    data = defaultdict(lambda: defaultdict(list))
    def generate_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """
    for row in tqdm.tqdm(dataset, desc=f'Processing {name}', disable=silent):
        prompt = generate_prompt(row['instruction'])
        data[prompt]['sft_target'] = f"{row['output']}"
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []
    return data

def get_humaneval(split: str,silent: bool = False,cache_dir: str = None,**kw):
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("openai_humaneval", split="test", cache_dir=cache_dir)
    data = defaultdict(lambda: defaultdict(list))

    def mk_prompt(task_id, prompt):
        # same prompt style your code uses elsewhere
        return f"""### Task {task_id}
{prompt}

### Response:
"""

    for row in tqdm.tqdm(ds, desc="Processing HumanEval", disable=silent):
        p = mk_prompt(row["task_id"], row["prompt"])
        # gold answer blob == json string produced by hf dataset
        data[p]["sft_target"] = row["canonical_solution"]      # ground-truth code
        data[p]["pairs"]      = []
        data[p]["responses"]  = []

    return data

def get_gsm8k(split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 1, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    dataset = load_dataset('gsm8k', 'main', split=split)
    num_conversations = len(dataset)
    dataset = dataset.select(range(int(num_conversations * data_fraction)))
    data = defaultdict(lambda: defaultdict(list))
    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
    for row in tqdm.tqdm(dataset, desc='Processing GSM8k', disable=silent):
        prompt = f"{row['question']}{QUESTION_PROMPT}"
        target = f"{row['answer']}".replace("####", ANSWER_PROMPT)
        data[prompt]['sft_target'] = target
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []
    return data


def get_commonsense(split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 1, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    dataset = load_dataset("zwhe99/commonsense_170k", split=split)
    num_conversations = len(dataset)
    dataset = dataset.select(range(int(num_conversations * data_fraction)))
    data = defaultdict(lambda: defaultdict(list))
    def generate_prompt(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Response:
                    """
    for row in tqdm.tqdm(dataset, desc='Processing CommonSense', disable=silent):
        prompt = generate_prompt(row['instruction'], row['input'])
        data[prompt]['sft_target'] = f"{row['output']}"
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []
    return data

def get_codealpaca(split: str, silent: bool = False, cache_dir: str = None, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=split)
    num_conversations = len(dataset)
    dataset = dataset.select(range(int(num_conversations * data_fraction)))
    data = defaultdict(lambda: defaultdict(list))
    def generate_prompt(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Response:
                    """
    for row in tqdm.tqdm(dataset, desc='Processing CodeAlpaca', disable=silent):
        prompt = generate_prompt(row['instruction'], row['input'])
        data[prompt]['sft_target'] = f"{row['output']}"
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []
    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, **kwargs):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    base_name, frac = _split_name_fraction(name)
    eff_fraction = frac if frac is not None else kwargs.get('data_fraction', 1.0)

    if base_name == 'gsm8k':
        data = get_gsm8k(split, silent=silent, cache_dir=cache_dir, data_fraction=eff_fraction)
    elif base_name == 'commonsense':
        data = get_commonsense(split, silent=silent, cache_dir=cache_dir, data_fraction=eff_fraction)
    elif base_name == 'codealpaca':
        if split == "test":
            data = get_humaneval(split="test", silent=silent, cache_dir=cache_dir)
        else:
            data = get_codealpaca(split=split, silent=silent, cache_dir=cache_dir, data_fraction=eff_fraction)
    elif base_name == "openai_humaneval" or base_name == "humaneval":
        data = get_humaneval(split, silent=silent, cache_dir=cache_dir, **kwargs)
    else:
        data = get_default(base_name, split, silent=silent, cache_dir=cache_dir, data_fraction=eff_fraction)

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
       ints [tokens] or strings [the original texts]) and returns a batch of examples,
       PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}

        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 64,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       **kwargs) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            base_name, _ = _split_name_fraction(name)  # --- NEW ---
            truncation_mode = 'keep_end' if base_name in ['hh', 'sharegpt', 'sharegpt4'] else 'keep_start'  # --- UPDATED ---
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, **kwargs).items():
                flat_data.append((
                    base_name,  # --- UPDATED: log base name, not "boolq:0.2" ---
                    prompt,
                    data['responses'],
                    data['pairs'],
                    data['sft_target'],
                    truncation_mode
                ))
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data)

        batch = []
        current_names = []
        for name, prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                current_names.append(name)
                example_idx += 1
                if len(batch) == batch_size:
                    # print which dataset this batch is primarily from
                    try:
                        maj = max(set(current_names), key=current_names.count)
                    except Exception:
                        maj = name
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True
                    batch = []
                    current_names = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    current_names.append(name)
                    example_idx += 1
                    if len(batch) == batch_size:
                        try:
                            maj = max(set(current_names), key=current_names.count)
                        except Exception:
                            maj = name
                        print(f"🟣 Now training on: {maj}")
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'Finished generating {n_examples} examples on {split} split')
                            done = True
                        batch = []
                        current_names = []

        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True


if __name__ == '__main__':
    import transformers
    cache_dir = os.path.join(os.getenv("HF_HOME", "~/.cache"), "datasets")
    tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    data_iterator_kwargs = dict(
        names=["gsm8k"],
        tokenizer=tokenizer,
        shuffle=True,
        max_length=512,
        max_prompt_length=256,
        sft_mode=True,
        prefs_path=None,
        num_turns=1,
        data_fraction=1,
    )
    iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=1, n_examples=100, batch_size=8, cache_dir=cache_dir)
    print(f'Loaded train data iterator')
    for batch in iterator:
        print(batch)
        break
