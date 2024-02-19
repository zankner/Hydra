# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

import os
from dataclasses import dataclass, field
import json
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from streaming import MDSWriter

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EVAL_SPLIT_SEED = 42

# Customized for training Hydra heads
class CustomizedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        data_ids = inputs["data_ids"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.args.hidden_state_offset != 0,
        )

        if self.args.hidden_state_offset == 0:
            base_hidden_states = outputs[0].clone()
        else:
            base_hidden_states = outputs[1][-(self.args.hidden_state_offset + 1)].clone()

        foo_loss = torch.zeros(input_ids.size(0), device=input_ids.device) 
        foo_logits = torch.zeros_like(base_hidden_states)

        data_ids = self.accelerator.gather(data_ids)
        input_ids = self.accelerator.gather(input_ids)
        labels = self.accelerator.gather(labels)
        attention_mask = self.accelerator.gather(attention_mask)
        all_base_hidden_states = self.accelerator.gather(base_hidden_states)

        if local_rank == 0:
            data_ids = data_ids.cpu().numpy()
            input_ids = input_ids.cpu().numpy()
            labels = labels.cpu().numpy()
            attention_mask = attention_mask.int().cpu().numpy()
            all_base_hidden_states = all_base_hidden_states.to(torch.float32).cpu().numpy()

            self.write_buffers["input_ids"][data_ids, ...] = input_ids
            self.write_buffers["labels"][data_ids, ...] = labels
            self.write_buffers["attention_mask"][data_ids, ...] = attention_mask
            self.write_buffers["base_hidden_states"][data_ids, ...] = all_base_hidden_states
        
        dist.barrier()

        return (foo_loss, foo_logits) if return_outputs else foo_loss
    

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Which data split to use."},
    )
    remote_upload_base: str = field(
        default=None,
        metadata={"help": "Remote location for training artifacts"}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    hidden_state_offset: int = field(
        default=0,
        metadata={"help": "Number of layers back from final layer to use as embeddings"},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}, {j}, {role}, {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            data_ids=i,
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            data_ids=np.array(i),
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    def read_jsonl_file(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                datum = json.loads(line)
                data.append(datum)
        return data

    data_json = read_jsonl_file(data_args.data_path)
    dataset = dataset_cls(data_json, tokenizer=tokenizer)

    return dataset


def build_hidden_states():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        device_map=f"cuda:{local_rank}"
    )

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    dataset = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Format write dir
    save_name = f"{model_args.model_name_or_path.split('/')[-1]}_off_{training_args.hidden_state_offset}"
    save_path = os.path.join(data_args.remote_upload_base, save_name, data_args.split)

    # Build streaming writer
    if local_rank == 0:
        os.makedirs(save_path, exist_ok=True)
        n_rows = len(dataset)
        input_ids_array = np.memmap(os.path.join(save_path, "input_ids.npy"), dtype=np.int64, mode="w+", shape=(n_rows, training_args.model_max_length))
        labels_array = np.memmap(os.path.join(save_path, "labels.npy"), dtype=np.int64, mode="w+", shape=(n_rows, training_args.model_max_length))
        attention_masks_array = np.memmap(os.path.join(save_path, "attention_masks.npy"), dtype=np.int64, mode="w+", shape=(n_rows, training_args.model_max_length))
        base_hidden_states_array = np.memmap(os.path.join(save_path, "base_hidden_states.npy"), dtype=np.float32, mode="w+", shape=(n_rows, training_args.model_max_length, config.hidden_size))
        write_buffers = {
            "input_ids": input_ids_array,
            "labels": labels_array,
            "attention_mask": attention_masks_array,
            "base_hidden_states": base_hidden_states_array,
        }
    else:
        write_buffers = None

    # import pdb; pdb.set_trace()
    # Start trainner
    training_args.remove_unused_columns = False
    trainer = CustomizedTrainer(
        model=model, tokenizer=tokenizer, args=training_args, 
    )
    trainer.write_buffers = write_buffers
    trainer.evaluate(dataset)

    # close the writer
    if local_rank == 0:
        input_ids_array.flush()
        del input_ids_array

        labels_array.flush()
        del labels_array

        attention_masks_array.flush()
        del attention_masks_array

        base_hidden_states_array.flush()
        del base_hidden_states_array
    

if __name__ == "__main__":
    build_hidden_states()
