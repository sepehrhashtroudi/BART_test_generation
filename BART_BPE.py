from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast
import pandas as pd
import os
import numpy as np
from datasets import load_dataset, load_metric
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


#df = pd.read_csv(path  + "_final.csv")
# train_dataset = load_dataset('csv', data_files="dataset/combined/train_combined.csv")
# eval_dataset = load_dataset('csv', data_files="dataset/combined/eval_combined.csv")
train_dataset = load_dataset('csv', data_files="dataset/train_final.csv")
eval_dataset = load_dataset('csv', data_files="dataset/eval_final.csv")
# train_dataset = load_dataset('csv', data_files="dataset/small/train_final_500.csv")
# eval_dataset = load_dataset('csv', data_files="dataset/small/eval_final_500.csv")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./BPE_tokenizer/BPE.json")
# train_dataset = create_dataset(df['0'].tolist(), df['0.1'].tolist(), tok, pad_truncate=True, max_length=128)
# eval_dataset = create_dataset(df['0'].tolist(), df['0.1'].tolist(), tok, pad_truncate=True, max_length=128)
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
                              "pad_token": "<pad>", "cls_token": "<s>"})
def tokenize_function(examples):
  
  model_inputs = tokenizer.batch_encode_plus(examples['0'], max_length=512, padding=True, truncation=True)

  with tokenizer.as_target_tokenizer():
      labels = tokenizer.batch_encode_plus(examples['0.1'], max_length=512, padding=True, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_train_dataset_word = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset_word = eval_dataset.map(tokenize_function, batched=True)
print(tokenized_train_dataset_word["train"])



metric = load_metric("./metric/sacrebleu.py")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


    
small_eval_dataset = tokenized_eval_dataset_word['train'].shuffle(seed=42).select(range(1000))
if os.path.exists("bart-large-cnn")==False:
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model.save_pretrained("./bart-large-cnn")
else:
    model = BartForConditionalGeneration.from_pretrained("./bart-large-cnn")
# tokenizer = ByteLevelBPETokenizer("/content/drive/MyDrive/colab_backup/BPE_pretrained_on_code/")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./BPE_tokenizer/BPE.json")
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
                              "pad_token": "<pad>", "cls_token": "<s>"})
training_args = Seq2SeqTrainingArguments(output_dir = "./saved_models/Bart_BPE_saved_models_full_tufano",\
                                         per_device_train_batch_size=2,\
                                         num_train_epochs=20,
                                         evaluation_strategy = "epoch",\
                                         per_device_eval_batch_size=2,\
                                         save_strategy='epoch',\
                                         load_best_model_at_end=True,\
                                         predict_with_generate=True)

data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )
# trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_train_dataset)
trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset_word['train'],
        eval_dataset=tokenized_eval_dataset_word['train'] ,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

trainer.train()
