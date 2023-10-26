import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tensorflow.python.ops.numpy_ops import np_config
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np_config.enable_numpy_behavior()
accuracy = evaluate.load("accuracy")
to_remove = [
    'text',
    'parent_id',
    'district',
    'districtId',
    'municipality',
    'message_id',
    'createdOn',
    'updatedOn']


# %%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def prepare_dataset():
    train_logdata = load_dataset("csv", data_files="data/output_enc_train.csv") \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    test_logdata = load_dataset("csv", data_files="data/output_enc_test.csv") \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    return train_logdata, test_logdata


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# %%

df = pd.read_csv("data/logs_25oct.csv")
df = df[['text', 'category']]
labels = list(df['category'].unique())
id2label = dict(zip(range(len(labels)), labels))
label2id = dict(zip(labels, range(len(labels))))


# %%

def get_training_args():
    return TrainingArguments(
        output_dir="model_run",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        do_train=True,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4
    )


# %%
def run_training(pre_model):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train, tokenized_test = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=14, id2label=id2label, label2id=label2id, trust_remote_code=True
    ).to(device)

    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_test["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


# %%
# mBERT
curr_model = "bert-base-multilingual-cased"
run_training(curr_model)

# %%
curr_model = "bert-base-cased"
run_training(curr_model)
# %%
curr_model = "ltg/norbert3-base"
run_training(curr_model)
