import pandas as pd
import torch
import os
from tqdm import tqdm
import random
import numpy as np
import time
import datetime

tqdm.pandas()
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-small', bos_token='<|bos|>',
                                          eos_token='<|eos|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2-small')
model.resize_token_embeddings(len(tokenizer))

model = torch.nn.DataParallel(model)

model = model.to(device)


class StorygenDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=1024):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.bos = "<|bos|>"
        self.pad = "<|pad|>"
        self.eos = "<|eos|>"

        self.src_input_ids = []
        self.attn_masks = []

        input_sentences_df = pd.read_csv(data_path, sep='\t', header=0)
        input_sentences_df = input_sentences_df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]

        for idx, src in tqdm(input_sentences_df.iterrows()):
            input_str = self.bos
            for s_id in range(1, 3):
                input_str += ' ' + src[f'sentence{s_id}'] + self.eos
            input_str += '\nTarget:'
            for s_id in range(3, 6):
                input_str += ' ' + src[f'sentence{s_id}'] + self.eos

            encodings_dict = self.tokenizer(input_str, truncation=True, max_length=max_length, padding="max_length")
            self.src_input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.src_input_ids)

    def __getitem__(self, item):
        return self.src_input_ids[item], self.attn_masks[item]


train_dataset = StorygenDataset(tokenizer, '../../data/storygen/train/train.tsv')
dev_dataset = StorygenDataset(tokenizer, '../../data/storygen/dev/dev.tsv')
test_dataset = StorygenDataset(tokenizer, '../../data/storygen/test/test.tsv')

batch_size = 8

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset), # Select batches randomly
            batch_size=batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            dev_dataset, # The validation samples.
            sampler=SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size=batch_size # Evaluate with this batch size.
        )


epochs = 2
learning_rate = 1e-6
warmup_steps = 1e2
epsilon = 1e-8

sample_every = 100

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

total_t0 = time.time()

training_stats = []


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        labels=b_labels,
                        attention_mask=b_masks,
                        # token_type_ids=None
                        )

        loss = outputs[0]

        loss = loss.mean()
        batch_loss = loss.item()

        total_train_loss += batch_loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            #                            token_type_ids=None,
                            attention_mask=b_masks,
                            labels=b_labels)

            loss = outputs[0]

        loss = loss.mean()
        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

output_dir = 'train_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv(output_dir + 'training_stats.csv', sep=',')

print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
