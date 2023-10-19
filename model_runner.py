import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("output.csv")
# train = int(len(df) * .8)
# test = int(len(df) * .2)
#
# train_df = df[:train]
# test_df = df[train:train + test]
df = df[['text', 'category']]
train_df, test_df = train_test_split(df, test_size=0.20)
labels = list(df['category'].unique())
id2label = dict(zip(range(len(labels)), labels))
label2id = dict(zip(labels, range(len(labels))))

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=14,
                                                           id2label=id2label, label2id=label2id)
training_args = TrainingArguments(output_dir="test_trained", per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8)
metric = evaluate.load("accuracy")


# %%

def preprocess(data):
    text = list(data['text'])
    category_list = list(data['category'])
    ins = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        out = tokenizer(category_list, padding="max_length", truncation=True, max_length= 14)

    data["input_ids"] = ins.input_ids
    data["attention_mask"] = ins.attention_mask
    data["decoder_input_ids"] = out.input_ids
    data["decoder_attention_mask"] = out.attention_mask
    data["labels"] = out.input_ids.copy()

    data["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                      data["labels"]]
    return data


# %%
train_ds = train_ds.map(preprocess, batched=True, remove_columns=['text', 'category', '__index_level_0__'])
train_ds.set_format(columns=["labels", "input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"])


# processed = preprocess(train_ds)


# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %%
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'],
                                            inputs['labels'])
        return (loss, outputs) if return_outputs else loss


trained = CustomTrainer(model=model,
                        args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=test_df,
                        compute_metrics=compute_metrics)

# %%
trained.train()
