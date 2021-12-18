import os

import torch
from datasets import load_dataset
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers import (
    BertTokenizerFast,
    AutoModel,
)

from _path import ROOT_PATH


def load_LM_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'Taiwan_news_dataset.py'))


def load_MLM_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset.py'))


def load_MLM_data_v2():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v2.py'))


def load_MLM_data_v3():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v3.py'))


def load_MLM_data_v4():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v4.py'))


def load_MLM_data_v5():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v5.py'))


def load_MLM_data_v6():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v6.py'))


def load_MLM_data_token():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_token.py'))


def load_MLM_NT_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_NT.py'))


def load_LM_NT_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script',
                     'Taiwan_news_dataset_notag.py'))


def load_tokenizer(tokenizer_name, max_length):
    tokenizer = BertTokenizerFast.from_pretrained('tokenizer/ckip')
    # Add special tokens.
    tokenizer.add_special_tokens({
        'pad_token':
        '[PAD]',
        'bos_token':
        '[CLS]',
        'sep_token':
        '[SEP]',
        'unk_token':
        '<unk>',
        'additional_special_tokens': [f'<per{i}>' for i in range(25)] +
        [f'<loc{i}>' for i in range(25)] + [f'<org{i}>' for i in range(25)]
    })
    tokenizer.model_max_length = max_length

    return tokenizer


def callate_fn_creater(tokenizer):
    # Set post processor to add special token.
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.get_vocab()["[CLS]"]),
            ("[SEP]", tokenizer.get_vocab()["[SEP]"]),
        ])

    def tokenizer_function(data):
        articles = [i['article'] for i in data]
        tokenized_result = tokenizer(articles,
                                     truncation=True,
                                     padding=True,
                                     return_tensors='pt')
        del tokenized_result['token_type_ids']

        return tokenized_result

    return tokenizer_function


def load_dataset_by_name(dataset_name: str):
    if dataset_name == 'Taiwan_news_dataset':
        return load_LM_data()
    elif dataset_name == 'MLM_dataset':
        return load_MLM_data()
    elif dataset_name == 'MLM_dataset_v2':
        return load_MLM_data_v2()
    elif dataset_name == 'MLM_dataset_v3':
        return load_MLM_data_v3()
    elif dataset_name == 'MLM_dataset_v4':
        return load_MLM_data_v4()
    elif dataset_name == 'MLM_dataset_v5':
        return load_MLM_data_v5()
    elif dataset_name == 'MLM_dataset_v6':
        return load_MLM_data_v6()
    elif dataset_name == 'MLM_dataset_token':
        return load_MLM_data_token()
    elif dataset_name == 'LM_NT_data':
        return load_LM_NT_data()
    elif dataset_name == 'MLM_NT_data':
        return load_MLM_NT_data()
    else:
        raise ValueError(f'Dataset name not exist {dataset_name}')


def create_data_loader(
    batch_size: int,
    tokenizer_name: str,
    max_length: int,
    dataset_name: str,
    shuffle: bool = True,
    testing: bool = False,
):
    # Load dataset.
    datasets = load_dataset_by_name(dataset_name=dataset_name)

    # Load tokenizer.
    tokenizer = load_tokenizer(tokenizer_name=tokenizer_name,
                               max_length=max_length)
    if testing:
        dataset = datasets['test']
    else:
        dataset = datasets['train']
    # Create data loader.
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=4,
                             collate_fn=callate_fn_creater(
                                 tokenizer=tokenizer, ))

    return data_loader


if __name__ == '__main__':
    data_loader = create_data_loader(2, 'chinese_tokenizer', max_length=128)
