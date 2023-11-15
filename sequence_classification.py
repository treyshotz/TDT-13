import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tensorflow.python.ops.numpy_ops import np_config
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, EvalPrediction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np_config.enable_numpy_behavior()
to_remove = [
    'text',
    'parent_id',
    'district',
    'districtId',
    'municipality',
    'message_id',
    'createdOn',
    'updatedOn']

training_file = "data/small.csv"
test_file = "data/small.csv"
# %%

label2id = {'Innbrudd': 0, 'Trafikk': 1, 'Brann': 2, 'Tyveri': 3, 'Ulykke': 4, 'Ro og orden': 5, 'Voldshendelse': 6,
            'Andre hendelser': 7, 'Savnet': 8, 'Skadeverk': 9, 'Dyr': 10, 'Sjø': 11, 'Redning': 12, 'Arrangement': 13}
id2label = {0: 'Innbrudd', 1: 'Trafikk', 2: 'Brann', 3: 'Tyveri', 4: 'Ulykke', 5: 'Ro og orden', 6: 'Voldshendelse',
            7: 'Andre hendelser', 8: 'Savnet', 9: 'Skadeverk', 10: 'Dyr', 11: 'Sjø', 12: 'Redning', 13: 'Arrangement'}
labels = list(label2id.keys())


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
    train_logdata = load_dataset("csv", data_files=training_file) \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    test_logdata = load_dataset("csv", data_files=test_file) \
        .map(preprocess_function, batched=True) \
        .remove_columns(to_remove)
    return train_logdata, test_logdata


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    inc = []
    pred = []
    for pos, it in enumerate(p.predictions):
        if p.predictions[pos].argmax() != p.label_ids[pos].argmax():
            inc.append(pos)
            pred.append(p.predictions[pos].argmax())

    test_df = pd.read_csv(test_file)
    test_df['predicted'] = pred
    test_df.iloc[inc].to_csv(f"incorrect_{curr_model.replace('/', '')}.csv")
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


# %%

def get_training_args(pre_model):
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
        save_total_limit=4,
        run_name=pre_model
    )


# %%
from torch import nn

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 4.0, 7.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def run_training(pre_model):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train, tokenized_test = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=14, id2label=id2label, label2id=label2id, trust_remote_code=True,
        problem_type="multi_label_classification"
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
    trainer.train()
    trainer.evaluate()
    trainer.model.save_pretrained(pre_model)

# %%
# mBERT
curr_model = "bert-base-multilingual-cased"
run_training(curr_model)

# # %%
# curr_model = "bert-base-cased"
# run_training(curr_model)
# # %%
# curr_model = "ltg/norbert3-base"
# run_training(curr_model)
