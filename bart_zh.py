from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BertTokenizer
from transformers import Trainer, TrainingArguments
import torch


tokenizer = BertTokenizer.from_pretrained('./fnlp_bart-large-chinese')
model = BartForConditionalGeneration.from_pretrained('./fnlp_bart-large-chinese')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
articles = ['北京是中国的首都']  # put your articles here
dct = tokenizer.batch_encode_plus(articles, max_length=1024, return_tensors="pt", pad_to_max_length=True)  # you can change max_length if you want

summaries = model.generate(
    input_ids=dct["input_ids"].to(device),
    attention_mask=dct["attention_mask"].to(device),
    num_beams=4,
    length_penalty=2.0,
    max_length=142,  # +2 from original because we start at step=1 and stop before max_length
    min_length=56,  # +1 from original because we start at step=1
    no_repeat_ngram_size=3,
    early_stopping=True,
    do_sample=False,
)  # change these arguments if you want

dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
print(dec)



train_texts, val_texts, train_labels, val_labels = train_test_split(df.articles, df.highlights, test_size=.2)
train_texts = train_texts.values.tolist()
train_labels = train_labels.values.tolist()
val_texts = val_texts.values.tolist()
val_labels = val_labels.values.tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_label_encodings = tokenizer(train_labels, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
val_label_encodings = tokenizer(val_labels, truncation=True, padding=True)


class PyTorchDatasetCreate(torch.utils.data.Dataset):
    def init(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PyTorchDatasetCreate(train_encodings, train_label_encodings)
val_dataset = PyTorchDatasetCreate(val_encodings, val_label_encodings)

training_args = TrainingArguments(
    output_dir= '',# output directory
    num_train_epochs=3, # total number of training epochs
    per_device_train_batch_size=1, # batch size per device during training
    per_device_eval_batch_size=1, # batch size for evaluation
    warmup_steps=200, # number of warmup steps for learning rate scheduler
    weight_decay=0.01, # strength of weight decay
    logging_dir='', # directory for storing logs
    logging_steps=10)

trainer = Trainer(
model=model, # the instantiated :hugs: Transformers model to be trained
args=training_args, # training arguments, defined above
train_dataset=train_dataset, # training dataset
eval_dataset=val_dataset) # evaluation dataset

trainer.train()
