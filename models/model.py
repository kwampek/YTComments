import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

special_tokens = {
    "additional_special_tokens": ["<META>", "<TEXT>", "<REPLY>"]
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token

META_ID = tokenizer.convert_tokens_to_ids("<META>")
TEXT_ID = tokenizer.convert_tokens_to_ids("<TEXT>")
REPLY_ID = tokenizer.convert_tokens_to_ids("<REPLY>")
NEW_LINE_ID = tokenizer.encode("\n", add_special_tokens=False)[0]

class SteroidDataset(Dataset):
    def __init__(self, df,
                 tokenizer,
                 max_title_tokens=20,
                 max_desc_tokens=50,
                 max_comment_tokens=50,
                 max_reply_tokens=50):

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        self.max_title = max_title_tokens
        self.max_desc = max_desc_tokens
        self.max_comment = max_comment_tokens
        self.max_reply = max_reply_tokens

        self.meta_id = META_ID
        self.text_id = TEXT_ID
        self.reply_id = REPLY_ID

        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    def _tok(self, text, max_len):
        text = str(text) if text else ""
        return self.tokenizer.encode(text, add_special_tokens=False)[:max_len]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = [self.meta_id, self.newline_id]

        def add(name, text, max_len):
            seq.extend(self.tokenizer.encode(f"{name}: ", add_special_tokens=False))
            seq.extend(self._tok(text, max_len))
            seq.append(self.newline_id)

        add("Video Title", row.get("video_title", ""), self.max_title)
        add("Video Description", row.get("video_description", ""), self.max_desc)
        add("Channel Title", row.get("channel_title", ""), self.max_title)
        add("Channel Description", row.get("channel_description", ""), self.max_desc)

        seq.append(self.text_id)
        seq.append(self.newline_id)
        seq.extend(self._tok(row.get("comment_text", ""), self.max_comment))
        seq.append(self.newline_id)

        seq.append(self.reply_id)
        seq.append(self.newline_id)
        seq.extend(self._tok(row.get("reply_text", ""), self.max_reply))

        input_ids = torch.tensor(seq, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        reply_pos = (input_ids == self.reply_id).nonzero(as_tuple=True)[0]

        if len(reply_pos) > 0:
            labels[:reply_pos[0] + 1] = -100
        else:
            labels[:] = -100

        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.df)



def prepare_meta(meta, comment=None, 
                 max_title_tokens=20,
                 max_desc_tokens=50,
                 max_comment_tokens=50,
                 max_reply_tokens=50):
    
    seq = [META_ID, NEW_LINE_ID]

    def add(name, text, max_len):
        seq.extend(tokenizer.encode(f"{name}: ", add_special_tokens=False))
        seq.extend(tokenizer.encode(text, add_special_tokens=False)[:max_len])
        seq.append(NEW_LINE_ID)

    add("Video Title", meta.get("video_title", ""), max_title_tokens)
    add("Video Description", meta.get("video_description", ""), max_desc_tokens)
    add("Channel Title", meta.get("channel_title", ""), max_title_tokens)
    add("Channel Description", meta.get("channel_description", ""), max_desc_tokens)

    seq.extend([TEXT_ID, NEW_LINE_ID])
    if comment is not None:    
        seq.extend(tokenizer.encode(comment))
        seq.extend([NEW_LINE_ID, REPLY_ID, NEW_LINE_ID])

    input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask


def collate_fn(batch):
    input_ids = pad_sequence([b[0] for b in batch],
                             batch_first=True,
                             padding_value=tokenizer.pad_token_id)

    attention_mask = pad_sequence([b[1] for b in batch],
                                  batch_first=True,
                                  padding_value=0)

    labels = pad_sequence([b[2] for b in batch],
                          batch_first=True,
                          padding_value=-100)

    return input_ids, attention_mask, labels


class SteroidGPT(nn.Module):
    def __init__(self, lora_config=None, first_rg_layer_num=30):
        super().__init__()

        self.tokenizer = tokenizer

        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.model.resize_token_embeddings(len(self.tokenizer))

        for i, block in enumerate(self.model.transformer.h):
            requires_grad = i >= first_rg_layer_num
            for param in block.parameters():
                param.requires_grad = requires_grad

        if lora_config:
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


    @torch.no_grad() 
    def generate_from_query(self, text, row, max_new_tokens=50, temperature=0.8, top_p=0.9): 
        self.model.eval() 
        device = next(self.model.parameters()).device 
        input_ids, attention_mask = SteroidDataset.prepare_meta(row, text) 
        input_ids = input_ids.to(device) 
        attention_mask = attention_mask.to(device) 

        return self.generate_from_batch(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_p=top_p )[0]


    @torch.no_grad()
    def generate_comment(self, meta, comment, max_new_tokens=100, temperature=0.8, top_p=0.9):
        self.model.eval()

        input_ids, attention_mask = prepare_meta(meta, comment)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id
        )

        generated = outputs[0][len(input_ids[0]):]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def create_model_and_optimizer(model_class, model_params, optim_params):
    model = model_class(**model_params).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        **optim_params
    )

    return model, optimizer


@st.cache_resource
def load_youcomment_model():
    model_path = "./models/weights.pt"

    checkpoint = torch.load(model_path, weights_only=False)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        bias="none"
    )

    model, _ = create_model_and_optimizer(
        SteroidGPT,
        {
            "lora_config": lora_config,
            "first_rg_layer_num": 36
        },
        {
            "lr": 1e-4
        }
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model