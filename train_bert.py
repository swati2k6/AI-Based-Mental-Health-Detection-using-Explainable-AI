from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np

def train_bert(X_train, X_test, y_train, y_test):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels.iloc[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, y_train)
    test_dataset = Dataset(test_encodings, y_test)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

    # ✅ PREDICTIONS (IMPORTANT FIX)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    return model, y_pred