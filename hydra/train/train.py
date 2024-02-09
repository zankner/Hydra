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

from dataclasses import dataclass, field
import json
import os
import math
import pathlib
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length, seed_worker, PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import LabelSmoother, nested_detach, find_batch_size, nested_concat, nested_numpify, IterableDatasetShard
from composer.utils import maybe_create_object_store_from_uri
from composer.utils.file_helpers import parse_uri
import wandb

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.nn import functional as F
from hydra.model.hydra_model import HydraModel, HydraConfig
from hydra.train.utils import get_scheduler

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EVAL_SPLIT_SEED = 42

# Customized for training Hydra heads
class CustomizedTrainer(Trainer):

    # Functions related to pre-computed data
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": False,
            "prefetch_factor": None if self.args.dataloader_num_workers == 0 else self.args.dataloader_prefetch_factor,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Metrics and loss functions
    def _score_preds(
        self,
        logits,
        labels,
        teacher_logits,
        teacher_labels,
        predict_hidden_states,
        label_hidden_states,
        lm_loss_fct,
        teacher_loss_fct,
        reconstruct_loss_fct,
        shift,
        log,
        log_key
    ):
        """
        Compute metrics such as acc and loss for given predictions.

        Args:
            logits (torch.Tensor): Predictions to compute metrics on.
            labels (torch.Tensor): True labels.
            loss_fct (torch.nn.Loss): Loss function to compute loss.
            log (dict): Dictionary to store computed metrics.
            log_key (str): Prefix for logging.
        Returns:
            torch.Tensor: The computed loss.
        """

        # Building LM terms
        hydra_logits = logits[:, : -shift].contiguous()
        teacher_logits = teacher_logits[:, shift - 1 : -1].contiguous()
        teacher_labels = teacher_labels[:, shift - 1 : -1].contiguous()
        hydra_labels = labels[..., shift :].contiguous()

        hydra_logits = hydra_logits.view(-1, hydra_logits.shape[-1])
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        teacher_labels = teacher_labels.view(-1)
        hydra_labels = hydra_labels.view(-1)

        hydra_labels = hydra_labels.to(hydra_logits.device)
        not_ignore_lm = hydra_labels.ne(IGNORE_TOKEN_ID)

        # Building reconstruction terms
        reconstruct_labels = labels[..., shift - 1 :].contiguous().view(-1)
        not_ignore_reconstruct = reconstruct_labels.ne(IGNORE_TOKEN_ID)

        # Hack for metrics on og states
        if shift - 1 == 0:
            hydra_pred_hidden_states = predict_hidden_states.contiguous()
        else:
            hydra_pred_hidden_states = predict_hidden_states[:, : -(shift - 1)].contiguous()
        hydra_label_hidden_states = label_hidden_states[:, shift - 1 :].contiguous()
        hydra_pred_hidden_states = hydra_pred_hidden_states.view(-1, hydra_pred_hidden_states.shape[-1])
        hydra_label_hidden_states = hydra_label_hidden_states.view(-1, hydra_label_hidden_states.shape[-1])

        # Compute losses
        cur_lm_loss = lm_loss_fct(hydra_logits, hydra_labels)
        cur_teacher_loss = teacher_loss_fct(hydra_logits[not_ignore_lm], teacher_logits[not_ignore_lm]) # When distilling, no ignore label
        cur_reconstruct_loss = reconstruct_loss_fct(hydra_pred_hidden_states[not_ignore_reconstruct], hydra_label_hidden_states[not_ignore_reconstruct])

        # Computing acc
        hydra_labels = hydra_labels[not_ignore_lm]
        teacher_labels = teacher_labels[not_ignore_lm]

        # Add top-k accuracy
        for k in range(1, 6):
            _, topk = hydra_logits.topk(k, dim=-1)
            topk = topk[not_ignore_lm]
            correct = topk.eq(hydra_labels.unsqueeze(-1)).any(-1)
            teacher_correct = topk.eq(teacher_labels.unsqueeze(-1)).any(-1)
            log[f"{log_key}_top{k}"] = correct.float().mean().item()
            log[f"{log_key}_teacher_top{k}"] = teacher_correct.float().mean().item()
        
        # logging losses
        log[f"{log_key}_lm_loss"] = cur_lm_loss.item()
        log[f"{log_key}_teacher_loss"] = cur_teacher_loss.item()
        log[f"{log_key}_reconstruct_loss"] = cur_reconstruct_loss.item()

        cur_loss = self.args.lm_loss_weight * cur_lm_loss + self.args.teacher_loss_weight * cur_teacher_loss + self.args.reconstruction_loss_weight * cur_reconstruct_loss
        log[f"{log_key}_loss"] = cur_loss.item()

        return cur_loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        return_log=False,
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

        # Unpack the data
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        base_hidden_states = None
        if "base_hidden_states" in inputs:
            base_hidden_states = inputs["base_hidden_states"].to(torch.bfloat16)

        # DDP will give us model.module
        if hasattr(model, "module"):
            hydra = model.module.hydra
        else:
            hydra = model.hydra

        all_hydra_logits, all_hydra_hidden_states, _, orig_logits, base_hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            base_hidden_states=base_hidden_states,
            run_hydra_head=True,
            output_orig=True,
            noise_alpha=self.args.noise_alpha if model.training else 0.0,
        )

        # Fix for smooth l1 loss
        all_hydra_logits = all_hydra_logits.to(torch.float32)
        all_hydra_hidden_states = all_hydra_hidden_states.to(torch.float32)
        orig_logits = orig_logits.to(torch.float32)
        base_hidden_states = base_hidden_states.to(torch.float32)

        # Get teacher probs
        teacher_probs = F.softmax(orig_logits, dim=-1)
        teacher_labels = teacher_probs.argmax(dim=-1)

        # Shift so that tokens < n predict n
        loss = 0
        lm_loss_fct = CrossEntropyLoss()
        teacher_loss_fct = CrossEntropyLoss()
        reconstruct_loss_fct = SmoothL1Loss()
        log = {}

        # Get base model perf
        _ = self._score_preds(
            orig_logits, 
            labels, 
            teacher_probs,
            teacher_labels,
            base_hidden_states, 
            base_hidden_states, 
            lm_loss_fct, 
            teacher_loss_fct,
            reconstruct_loss_fct, 
            1, 
            log, 
            "orig"
        )

        for i in range(hydra):
            shift = 2 + i
            loss += self._score_preds(
                all_hydra_logits[i],
                labels,
                teacher_probs,
                teacher_labels,
                all_hydra_hidden_states[i],
                base_hidden_states,
                lm_loss_fct,
                teacher_loss_fct,
                reconstruct_loss_fct,
                shift,
                log,
                f"hydra{i}"
            )

        self.log(log)

        if return_log:
            return (loss, all_hydra_logits, log)
        return (loss, all_hydra_logits) if return_outputs else loss
    
    # Below is overwriting evals so we can get the specific metrics
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs, log = self.compute_loss(model, inputs, return_outputs=True, return_log=True)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            return (loss, log, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, log, logits, None)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix= "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (self.accelerator.prepare_model(model, evaluation_mode=True))

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

        batch_size = self.args.eval_batch_size

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        logs_host = {}

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_logs = {}
        # Will be useful when we have an iterable dataset so don't know its length.

        # Define gather
        self.gather_function = self.accelerator.gather_for_metrics

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, log, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if log is not None:
                for key in list(log.keys()):
                    logs = self.gather_function((torch.tensor([log[key]], device = loss.device).repeat(batch_size)))
                    logs_host[key] = logs if key not in logs_host else nested_concat(logs_host[key], logs, padding_index = -100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if logs_host is not None:
            for key in logs_host.keys():
                logs = nested_numpify(logs_host[key])
                all_logs[key] = np.nanmean(logs).item()
        
        metrics = all_logs

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = np.nanmean(all_losses).item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        #return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=None)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)
    
    # Overwriting save to only save the hydra heads
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{round(self.state.epoch)}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
        self.save_model(output_dir, _internal_call=True)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.hydra_head.state_dict(),
            os.path.join(output_dir, "hydra_lm_head.pt"),
        )
        self.args.hydra_config.save_pretrained(output_dir)
    
    # Overwriting scheduler so final lr can be specified
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            assert self.args.final_lr_multiplier < 1.0 + 1e-10, "final_lr_multiplier must be less than 1.0"
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                final_lr_multiplier=self.args.final_lr_multiplier,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    

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
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    precomputed_data: bool = field(default=False, metadata={"help": "Whether the hidden states are precomputed"})
    lazy_preprocess: bool = True
    eval_split_ratio: float = field(default=0.05, metadata={"help": "train / eval split ratio."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    hydra_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Hydra heads."},
    )
    hydra_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Hydra head."},
    )
    hydra_head_arch: str = field(
        default="mlp",
        metadata={"help": "What model architecture to use for Hydra heads."},
    )
    grounded_heads: bool = field(
        default=True,
        metadata={"help": "Whether to ground the Hydra heads on previous head predictions."},
    )
    hidden_state_offset: int = field(
        default=0,
        metadata={"help": "Number of layers back from final layer to use as embeddings"},
    )
    remote_upload_base: str = field(
        default=None,
        metadata={"help": "Remote location for training artifacts"}
    )
    dataloader_prefetch_factor: int = field(
        default=2,
    )
    global_batch_size: int = field(
        default=32,
        metadata={"help": "Global batch size."},
    )
    final_lr_multiplier: float = field(
        default=0.0,
        metadata={"help": "Final learning rate multiplier."},
    )
    noise_alpha: int = field(
        default=0,
        metadata={"help": "Noise std for input."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate."},
    )
    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for LM loss"},
    )
    teacher_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for teacher distilation loss"},
    )
    reconstruction_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for reconstruction loss"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_model(lm_head: torch.nn.Module, output_dir: str):
    """
    Save LM heads to potentially remote object store

    Args:
        lm_head (torch.Module): The LM head module.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = lm_head.state_dict()
    object_store = maybe_create_object_store_from_uri(output_dir)
    if object_store is None:
        torch.save(state_dict, os.path.join(output_dir, "hydra_lm_head.pt"))
    else:
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(state_dict, tmp)
            object_store.upload_object("hydra_lm_head.pt", tmp.name)

def remote_upload_model(training_dir: str, upload_dir: str):
    """
    Push model to remote object store

    Args:
        training_dir (str): The directory the model was stored to during training.
        upload_dir (str): The remote directory the model should be uploaded to.
    """
    object_store = maybe_create_object_store_from_uri(upload_dir)
    if object_store is None:
        print(f"Tried to create remote object store at: {upload_dir}, but failed")
    else:
        upload_past_bucket = parse_uri(upload_dir)[-1]
        print(f"Saving training artifacts to remote: {upload_dir}")
        for root, dirs, files in os.walk(training_dir):
            for file in files:
                load_path = os.path.join(root, file)
                upload_path = os.path.join(upload_past_bucket, *load_path.split(os.sep)[1:])
                print(f"Uploading: {upload_path}")
                object_store.upload_object(upload_path, load_path)


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
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

class PrecomputedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, data_root, model_max_length, hidden_size):
        super(PrecomputedDataset, self).__init__()

        rank0_print("Loading precomputed data")
        self.input_ids_array = np.memmap(os.path.join(data_root, "input_ids.npy"), dtype=np.int64, mode="r")
        self.labels_array = np.memmap(os.path.join(data_root, "labels.npy"), dtype=np.int64, mode="r")
        self.attention_masks_array = np.memmap(os.path.join(data_root, "attention_masks.npy"), dtype=np.int64, mode="r")
        self.base_hidden_states_array = np.memmap(os.path.join(data_root, "base_hidden_states.npy"), dtype=np.float32, mode="r")

        self.input_ids_array = self._reshape_memmap(self.input_ids_array, (model_max_length,))
        self.labels_array = self._reshape_memmap(self.labels_array, (model_max_length,))
        self.attention_masks_array = self._reshape_memmap(self.attention_masks_array, (model_max_length,))
        self.base_hidden_states_array = self._reshape_memmap(self.base_hidden_states_array, (model_max_length, hidden_size))
    
    def _reshape_memmap(self, array, final_dims):
        total_el = array.shape[0]
        n_rows = int(total_el / np.prod(final_dims))
        return array.reshape(n_rows, *final_dims)

    def __len__(self):
        return self.input_ids_array.shape[0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ret = dict(
            input_ids=self.input_ids_array[i],
            labels=self.labels_array[i],
            attention_mask=self.attention_masks_array[i],
            base_hidden_states=self.base_hidden_states_array[i]
        )
        return ret

def read_jsonl(path: str) -> Sequence[Dict]:
    """Read a JSONL file.

    Args:
        path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries.
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def make_raw_supervised_data_module(
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

    train_json = read_jsonl(os.path.join(data_args.data_path, "train.json"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    eval_json = read_jsonl(os.path.join(data_args.data_path, "val.json"))
    eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def make_precomputed_supervised_data_module(data_args, training_args, hidden_size):
    train_dataset = PrecomputedDataset(
        data_root=os.path.join(data_args.data_path, "train"),
        model_max_length=training_args.model_max_length,
        hidden_size=hidden_size
    )
    eval_dataset = PrecomputedDataset(
        data_root=os.path.join(data_args.data_path, "val"),
        model_max_length=training_args.model_max_length,
        hidden_size=hidden_size
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def make_supervised_data_module(tokenizer, data_args, training_args, hidden_size):
    if data_args.precomputed_data:
        return make_precomputed_supervised_data_module(data_args, training_args, hidden_size)
    else:
        return make_raw_supervised_data_module(tokenizer, data_args)

def train():
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

    # Add Hydra heads
    hydra_lm_head = HydraModel(
        model,
        hydra_num_heads=training_args.hydra_num_heads,
        hydra_num_layers=training_args.hydra_num_layers,
        hydra_head_arch=training_args.hydra_head_arch,
        base_model_name_or_path=model_args.model_name_or_path,
        grounded_heads=training_args.grounded_heads,
        hidden_state_offset=training_args.hidden_state_offset,
        dropout_rate=training_args.dropout_rate,
    )

    # Removing model if pre-computed
    if data_args.precomputed_data:
        del model
        hydra_lm_head.base_model = None
    
    # Format output dir
    readable_arch_map = {
        "mlp": "mlp",
        "prefix-mlp": "pmlp",
        "cross-attn": "ca",
        "eagle-attn": "ea",
    }
    base_run_name = f"{'ground' if training_args.grounded_heads else 'spec'}_{readable_arch_map[training_args.hydra_head_arch]}_{model_args.model_name_or_path.split('/')[-1]}_nh_{training_args.hydra_num_heads}_nl_{training_args.hydra_num_layers}_ep_{int(training_args.num_train_epochs)}_bs_{training_args.global_batch_size}_lr_{training_args.learning_rate}_lrf_{training_args.final_lr_multiplier}_ws_{training_args.warmup_steps}_off_{training_args.hidden_state_offset}_n_{training_args.noise_alpha}_lw_{training_args.lm_loss_weight}_tw_{training_args.teacher_loss_weight}_rw_{training_args.reconstruction_loss_weight}"
    save_run_name = f"{base_run_name}_sd_{training_args.seed}"
    training_args.output_dir = os.path.join(training_args.output_dir, save_run_name)
    print("Saving ckpt locally to: ", training_args.output_dir)

    
    # Creating wandb logger
    training_args.report_to = ["wandb"]
    if local_rank == 0:
        wandb.init(project="hydra-decoding", 
            name=save_run_name,
            group=base_run_name,
            config={"train_args": training_args, "model_args": model_args, "data_args": data_args},
        )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args, hidden_size=config.hidden_size)

    # Generate Hydra config for pushing to HF hub
    hydra_config = HydraConfig(
        hydra_num_heads=training_args.hydra_num_heads,
        hydra_num_layers=training_args.hydra_num_layers,
        hydra_head_arch=training_args.hydra_head_arch,
        base_model_name_or_path=model_args.model_name_or_path,
        grounded_heads=training_args.grounded_heads,
        hidden_state_offset=training_args.hidden_state_offset,
    )

    # Save Hydra config
    hydra_config.save_pretrained(training_args.output_dir)
    training_args.hydra_config = hydra_config # For saving during checkpointing

    # import pdb; pdb.set_trace()
    # Start trainner
    training_args.remove_unused_columns = False
    trainer = CustomizedTrainer(
        model=hydra_lm_head, tokenizer=tokenizer, args=training_args, **data_module 
    )

    trainer.train()

    #####
    ## Commented out in official code base
    ####
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Save HydraHead seperately
    if hasattr(hydra_lm_head, "module"):
        lm_head = hydra_lm_head.module.hydra_head
    else:
        lm_head = hydra_lm_head.hydra_head

    # Save Hydra heads
    torch.save(
        lm_head.state_dict(),
        os.path.join(training_args.output_dir, "hydra_lm_head.pt"),
    )
    if training_args.remote_upload_base is not None and local_rank == 0:
        remote_upload_model(training_args.output_dir, training_args.remote_upload_base)


if __name__ == "__main__":
    train()
