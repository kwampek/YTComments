import streamlit as st

st.set_page_config(page_title="Paper", layout="wide")

st.title("YouComments")


st.markdown("""
<div style="background-color:#0e1117;padding:20px;border-radius:12px;border:1px solid #262730">

<h2 style="color:#4CAF50;">🧠 Instruction Finetuning для GPT-2 Large + LoRA</h2>

<p style="font-size:16px;line-height:1.6;">
Базовая модель <b>GPT-2 Large</b> была дообучена для корректного восприятия инструкций 
и генерации <b>комментариев и ответов</b> к ним.
</p>

Во время использования обращается к YouTube, рекомендую включить vpn. 😭

<hr style="border:1px solid #262730;">

<h4>📌 Формат входных данных</h4>

<pre style="background-color:#161a23;padding:15px;border-radius:8px;color:#e6edf3;">
&lt;META&gt;
Video Title: 
Video Description:
Channel Title:
Channel Description:

&lt;TEXT&gt;
Comment

&lt;REPLY&gt;
Reply
</pre>

<h4>⚙️ Как это работает</h4>

<ul style="line-height:1.6;">
<li>Мета-информация о видео добавляется прямо в запрос</li>
<li>Модель учитывает контекст видео при генерации ответа</li>
<li>Можно строить <b>диалоги любой длины</b>, рекурсивно подставляя ответы</li>
</ul>

<p style="background-color:#161a23;padding:12px;border-radius:8px;">
💡 <b>Идея:</b> заменяя <code>Comment</code> на предыдущий <code>Reply</code>, 
можно генерировать цепочку диалога не теряя информации о видео.  
</p>

</div>
""", unsafe_allow_html=True)


with st.expander("📦 Код модели"):
    st.code("""
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

class SteroidGPT(nn.Module):
    def __init__(self, lora_config=None, first_rg_layer_num=18, hidden_size=1280):
        super().__init__()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.model.resize_token_embeddings(len(self.tokenizer))

        for i, block in enumerate(self.model.transformer.h):
            if i < first_rg_layer_num:
                for param in block.parameters():
                    param.requires_grad = False

        for i, block in enumerate(self.model.transformer.h):
            if i >= first_rg_layer_num:
                for param in block.parameters():
                    param.requires_grad = True

        if lora_config:
            self.model = get_peft_model(self.model, lora_config)

        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    @torch.no_grad()
    def generate_from_batch(self, input_ids, attention_mask, max_new_tokens=200, temperature=0.8, top_p=0.9):
        self.model.eval()
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id
        )

        generated_only = generated_ids[:, input_ids.size(1):]
        decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_only]
        return decoded
""", language="python")



st.markdown("""
<div style="background-color:#0e1117;padding:22px;border-radius:14px;border:1px solid #262730">

<h2 style="color:#4CAF50;">📦 Dataset Overview</h2>

<p style="font-size:16px;line-height:1.6;">
Датасет автоматически собирается с <b>YouTube</b> и предназначен для обучения моделей,
генерирующих <b>ответы на комментарии</b> с учетом контекста видео.
</p>

<hr style="border:1px solid #262730;">

<h3>🌍 Источник данных</h3>
<ul style="line-height:1.7;">
<li>🔥 Trending-видео (регион: <b>US</b>)</li>
<li>💬 Комментарии и ответы (reply threads)</li>
</ul>

<hr style="border:1px solid #262730;">

<h3>🧾 Содержимое</h3>

<ul style="line-height:1.7;">
<li>💬 Тексты комментариев</li>
<li>↩️ Ответы на комментарии</li>
<li>👍 Количество лайков</li>
<li>👤 Автор комментария</li>
<li>🎬 Метаданные видео</li>
<li>📺 Метаданные канала</li>
</ul>

<hr style="border:1px solid #262730;">

<h3>🧹 Предобработка</h3>

<ul style="line-height:1.7;">
<li>🇬🇧 Оставлен только <b>английский язык</b></li>
<li>🚫 Удалены эмодзи и шум</li>
<li>✂️ Ограничение длины текста (token truncation)</li>
</ul>

<hr style="border:1px solid #262730;">

<h3>📏 Размер датасета</h3>

<div style="display:flex;gap:20px;">
    <div style="background-color:#161a23;padding:15px;border-radius:10px;flex:1;text-align:center;">
        <h4>Train</h4>
        <p style="font-size:20px;"><b>18,830</b></p>
    </div>
    <div style="background-color:#161a23;padding:15px;border-radius:10px;flex:1;text-align:center;">
        <h4>Test</h4>
        <p style="font-size:20px;"><b>4,708</b></p>
    </div>
</div>

<hr style="border:1px solid #262730;">

<h3>🧠 Формирование входа модели</h3>

<p style="line-height:1.6;">
Каждый пример преобразуется в единую последовательность с явной структурой:
</p>

<pre style="background-color:#161a23;padding:15px;border-radius:8px;color:#e6edf3;">
&lt;META&gt;
Video Title: ...
Video Description: ...
Channel Title: ...
Channel Description: ...

&lt;TEXT&gt;
Comment

&lt;REPLY&gt;
Reply
</pre>

<p style="background-color:#161a23;padding:12px;border-radius:8px;">
💡 Модель обучается предсказывать <b>только часть &lt;REPLY&gt;</b>, 
игнорируя остальной контекст (labels = -100)
</p>

<hr style="border:1px solid #262730;">

<h3>⚙️ Особенности Dataset-класса</h3>

<ul style="line-height:1.7;">
<li>📏 Ограничение длины для каждого поля (title, description, comment, reply)</li>
<li>🔤 Токенизация без добавления специальных токенов</li>
<li>🧩 Явное разделение блоков через <code>&lt;META&gt;</code>, <code>&lt;TEXT&gt;</code>, <code>&lt;REPLY&gt;</code></li>
<li>🎯 Маскирование loss — обучение только на reply</li>
</ul>

</div>
""", unsafe_allow_html=True)




with st.expander("Dataset"):
    st.code("""
special_tokens = {
    "additional_special_tokens": ["<META>", "<TEXT>", "<REPLY>"]
}

import torch
from torch.utils.data import Dataset

class SteroidDataset(Dataset):
    def __init__(self, df, tokenizer,
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

        self.meta_id = tokenizer.convert_tokens_to_ids("<META>")
        self.text_id = tokenizer.convert_tokens_to_ids("<TEXT>")
        self.reply_id = tokenizer.convert_tokens_to_ids("<REPLY>")
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.colon_id = tokenizer.encode(":", add_special_tokens=False)[0]

    def _tokenize_and_truncate(self, text, max_len):
        text = str(text) if text else ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens[:max_len]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = [self.meta_id, self.newline_id]

        def add_field(name, text, max_len):
            seq.extend(self.tokenizer.encode(f"{name}:", add_special_tokens=False))
            seq.append(self.colon_id)
            seq.extend(self._tokenize_and_truncate(text, max_len))
            seq.append(self.newline_id)

        add_field("Video Title", row.get('video_title', ""), self.max_title)
        add_field("Video Description", row.get('video_description', ""), self.max_desc)
        add_field("Channel Title", row.get('channel_title', ""), self.max_title)
        add_field("Channel Description", row.get('channel_description', ""), self.max_desc)

        seq.append(self.text_id)
        seq.append(self.newline_id)
        seq.extend(self._tokenize_and_truncate(row.get('comment_text', ""), self.max_comment))
        seq.append(self.newline_id)

        seq.append(self.reply_id)
        seq.append(self.newline_id)
        seq.extend(self._tokenize_and_truncate(row.get('reply_text', ""), self.max_reply))

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
""", language="python")

st.markdown("""
<div style="background-color:#0e1117;padding:22px;border-radius:14px;border:1px solid #262730">

<h2 style="color:#4CAF50;">🚀 Обучение модели</h2>

<p style="font-size:16px;line-height:1.6;">
Данный пайплайн реализует <b>гибкий и устойчивый цикл обучения</b> модели с поддержкой:
обучения, валидации, сохранения чекпоинтов и визуализации метрик в реальном времени. Взят из задачи image_captioning.
</p>


</div>
""", unsafe_allow_html=True)



st.header("📌 Итог")

st.markdown("""
<div style="background-color:#0e1117;padding:22px;border-radius:14px;border:1px solid #262730">

<h2 style="color:#4CAF50;">🏁 Результаты обучения</h2>

<p style="font-size:16px;line-height:1.7;">
В ходе эксперимента была успешно <b>дообучена модель</b> с использованием подхода LoRA.
</p>

<hr style="border:1px solid #262730;">

<h3>📊 Ключевые наблюдения</h3>

<ul style="line-height:1.8;">
<li>📉 <b>Loss значительно снизился</b> в процессе обучения</li>
<li>📊 <b>Accuracy не показала существенного роста</b> относительно базовой модели</li>
</ul>

<hr style="border:1px solid #262730;">

<h3>🧠 Интерпретация</h3>

<p style="line-height:1.7;">
Отсутствие заметного роста accuracy является <b>ожидаемым результатом</b>, учитывая:
</p>

<ul style="line-height:1.8;">
<li>⏱️ Ограниченное количество эпох обучения</li>
<li>📦 Небольшой размер датасета</li>
<li>🪶 Использование <b>LoRA</b> (обучается лишь малая часть параметров)</li>
</ul>

<p style="background-color:#161a23;padding:12px;border-radius:8px;">
💡 Несмотря на это, снижение loss указывает на то, что модель <b>лучше адаптировалась к формату задачи</b>
и начала увереннее предсказывать целевую последовательность
</p>

<hr style="border:1px solid #262730;">

<h3>🚀 Вывод</h3>

<p style="line-height:1.7;">
Полученные результаты демонстрируют <b>корректную работу пайплайна обучения</b> 
и подтверждают, что даже при ограниченных ресурсах модель способна 
адаптироваться к новой задаче генерации.
</p>

</div>
""", unsafe_allow_html=True)