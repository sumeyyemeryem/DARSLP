import torch
import importlib
from collections import Counter
import yaml

def load_config(path="train.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def create_mask(seq_lengths, max_len, device="cpu"):
    # mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device).clone().detach()[:, None]
    # mask = torch.arange(max_len, device=device)[None, :] < seq_lengths.clone().detach()[:, None]
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device).clone().detach()[:, None]
    return mask.bool()



def add_noise(x, noise_std=0.5):
    x_float = x.float()  # Convert to floating-point data type
    noise = torch.randn_like(x_float)
    return x_float + noise * noise_std


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key "target" to instatiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_vocab(file_paths, min_freq=1):
    data = []
    for path in file_paths:
        with open(path, "r", encoding='utf-8') as file:
            for line in file:
                data.append(line.strip())
    vocab = build_vocab(data, min_freq)
    return vocab

def build_vocab(data, min_freq):
    """
    Build a vocabulary from the given data.

    Args:
        data (list[str]): A list of strings, where each string represents a line of data.
        min_freq (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        dict: A dictionary mapping tokens to their indices in the vocabulary.
    """
    tokens = [token for line in data for token in line.split()]
    token_counts = Counter(tokens)
    token_list = [token for token, count in token_counts.items() if count >= min_freq]

    # Add special tokens to the vocabulary
    special_tokens = ['<sos>', '<eos>', '<pad>']
    vocab = {token: idx for idx, token in enumerate(special_tokens + token_list)}

    return vocab


def strings_to_indices(string, vocab):
    tokens = string.split()
    indices = [vocab["<sos>"]] + [vocab[token] for token in tokens if token in vocab] + [vocab["<eos>"]]
    return indices


def indices_to_strings(indices_list, vocab, sos_token='<sos>', eos_token='<eos>', pad_token='<pad>'):
    """
    Convert a list of lists of indices to a list of strings based on the given vocabulary.

    Args:
        indices_list (list[list[int]]): The input list of lists of indices to be converted.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        list[str]: A list of strings representing the input indices.
    """
    inv_vocab = {idx: token for token, idx in vocab.items()}
    special_indices = [vocab[token] for token in [sos_token, eos_token, pad_token]]
    strings = []
    for indices in indices_list:
        indices = indices.tolist()
        tokens = [inv_vocab[idx] for idx in indices if idx not in special_indices]
        string = ' '.join(tokens)
        strings.append(string)
    return strings