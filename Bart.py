

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
path = "eval"
df = pd.read_csv( path  + "_final.csv")
print(df['0.1'])
dataset = load_dataset('csv', data_files=  path  + "_final.csv")
model = BartForConditionalGeneration.from_pretrained("./distilbart-cnn-12-6", forced_bos_token_id=0)
#model.save_pretrained("./distilbart-cnn-12-6")
tokenizer = BartTokenizer.from_pretrained("./distilbart-cnn-12-6/tokenizer")
#tokenizer.save_pretrained("./distilbart-cnn-12-6/tokenizer")
# train_dataset = create_dataset(df['0'].tolist(), df['0.1'].tolist(), tok, pad_truncate=True, max_length=128)
# eval_dataset = create_dataset(df['0'].tolist(), df['0.1'].tolist(), tok, pad_truncate=True, max_length=128)
def tokenize_function(examples):
  model_inputs = tokenizer.batch_encode_plus(examples['0'], max_length=512, padding=True, truncation=True)

  with tokenizer.as_target_tokenizer():
      labels = tokenizer.batch_encode_plus(examples['0.1'], max_length=512, padding=True, truncation=True)

  # def tokenize_function(examples):
  #     return tok(examples['0'],examples['0.1'], padding="max_length", truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)



from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
small_train_dataset = tokenized_datasets['train']



training_args = Seq2SeqTrainingArguments(output_dir = "saved_models",per_device_train_batch_size=4)

data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
# trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_train_dataset)
trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_train_dataset ,
        tokenizer=tokenizer,
        data_collator=data_collator
        
        
    )

trainer.train()
