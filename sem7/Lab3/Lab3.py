import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

import urllib.request


with open('prestuplenie_i_nakazanie.txt', 'r', encoding='utf-8') as f:
     full_text = f.read()


# Предобработка текста:
# 1. Убираем лишние пробелы и переносы строк
# 2. Разбиваем на абзацы (каждый абзац — отдельный пример)
def clean_and_split(text, min_length=50):
    # Заменяем множественные переносы на разделитель абзацев
    text = re.sub(r'\n{2,}', '\n', text)
    # Разбиваем по одиночным переносам
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) >= min_length]
    return paragraphs

raw_texts = clean_and_split(full_text, min_length=80)
print(f"Всего абзацев в датасете: {len(raw_texts)}")
print(f"Пример первого абзаца:\n{raw_texts[0][:200]}...")


# Создаем Dataset
dataset = Dataset.from_dict({"text": raw_texts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Загружаем токенизатор от ruGPT-2 (поддерживает кириллицу)
model_name = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Для Достоевского характерны длинные предложения — увеличим max_length
MAX_LENGTH = 128

def tokenize_function(examples):
    # Токенизируем текст
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_special_tokens_mask=False
    )
    # копируем input_ids в labels
    # Для GPT-моделей labels = input_ids (модель сама делает сдвиг внутри)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Обучающая выборка: {len(tokenized_datasets['train'])} примеров")
print(f"Тестовая выборка: {len(tokenized_datasets['test'])} примеров")

model = GPT2LMHeadModel.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./dostoevsky_rugpt",
    per_device_train_batch_size=2,    # Уменьшено из-за большой длины последовательности
    per_device_eval_batch_size=2,
    num_train_epochs=5,               # 5 эпох достаточно для стилизации
    learning_rate=3e-5,               # Меньше LR — стабильнее обучение на литературе
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    fp16=torch.cuda.is_available()    # Смешанная точность для ускорения
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

print("Начинаем дообучение на романе Достоевского...")
trainer.train()
print("Обучение завершено!")

def generate(prompt, max_new_tokens=150, temperature=0.85, top_k=50, top_p=0.92):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompts = [
    "Раскольников вышел из своей каморки и",
    "Старуха процентщица сидела за столом и",
    "Соня Мармеладова тихо сказала:",
    "В темной петербургской ночи Раскольников думал о",
    "— Позвольте вас спросить, — начал Раскольников,"
]

print("\n" + "=" * 60)
print("ГЕНЕРАЦИЯ ТЕКСТА В СТИЛЕ ДОСТОЕВСКОГО")
print("=" * 60)

for i, p in enumerate(prompts, 1):
    print(f"\n[Промпт {i}]: {p}")
    print("-" * 60)
    result = generate(p)
    print(result)
    print("=" * 60)


prompt = "Раскольников вошёл в трактир и"

print("Greedy (жадный поиск, temperature=1.0, do_sample=False):")
print(generate.__wrapped__ if hasattr(generate, '__wrapped__') else None)

# Жадный поиск
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))

print("\nНизкая температура (0.3) — консервативно:")
print(generate(prompt, temperature=0.3))

print("\nВысокая температура (1.2) — креативно:")
print(generate(prompt, temperature=1.2))

print("\nTop-k = 10 — ограниченный словарь:")
print(generate(prompt, top_k=10))