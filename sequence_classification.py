import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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

df = pd.read_csv("data/logs_25oct.csv")
df = df[['text', 'category']]
labels = list(df['category'].unique())
id2label = dict(zip(range(len(labels)), labels))
label2id = dict(zip(labels, range(len(labels))))


# %%
def preprocess_function(examples):
    tokenized_text = tokenizer(examples["text"], truncation=True)
    labels_batch = examples['label']
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(tokenized_text.encodings), len(labels)))
    # fill numpy array
    for pos, obj in enumerate(labels_batch):
        labels_matrix[pos, obj] = 1

    tokenized_text["label"] = labels_matrix.tolist()

    return tokenized_text


def prepare_dataset():
    train_logdata = load_dataset("csv", data_files="data/output_enc_train.csv") \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    test_logdata = load_dataset("csv", data_files="data/output_enc_test.csv") \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    return train_logdata, test_logdata


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    # pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# %%

def get_training_args(pre_model):
    return TrainingArguments(
        output_dir="model_run",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        do_train=True,
        warmup_steps=200,
        fp16=True,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4,
        run_name=pre_model
    )


# %%

def collect_wrong_eval(trainer: Trainer):
    _, tokenized_test = prepare_dataset()
    for it in enumerate(tokenized_test['train']):
        trainer.model()
        encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}


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
        args=get_training_args(pre_model),
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_test["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.model()

    trainer.train()
    trainer.evaluate()

    # collect_wrong_eval(trainer)


# %%

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
