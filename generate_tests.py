# from datasets import load_dataset, load_metric
# from transformers import AutoTokenizer
# from transformers import TrainingArguments
# from transformers import Trainer
# from transformers import BartForConditionalGeneration, BartTokenizer
# from transformers import RobertaTokenizerFast
# from transformers import PreTrainedTokenizerFast
# import pandas as pd
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_tokenizer/BPE.json")
# tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
#                               "pad_token": "<pad>", "cls_token": "<s>"})
# # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = BartForConditionalGeneration.from_pretrained("Bart_BPE_saved_models/checkpoint-390015")

# dataset = load_dataset('csv', data_files="dataset/" + 'eval'  + "_final.csv")
# text = dataset["train"]['0'][0:3]
# # print(text)

# tokenized_text = tokenizer.batch_encode_plus(text, max_length=512, padding=True, truncation=True,return_tensors='pt')
# # print(tokenized_text)
# translation = []
# for i in tokenized_text:
#     print(i)
#     print("next######################################################################")
#     # translation.append(model.generate(i))
# # # print(translation)
# # translation = model.generate(**tokenized_text)
# # print(translation)
# translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

# print(text)
# with open("generated_tests.tests", "w") as tests:
#     tests.write(translated_text)
# # Print translated text
# print(translated_text)
# # Perform translation and decode the output

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
# train_dataset = load_dataset('csv', data_files="dataset/train_final.csv")
eval_dataset = load_dataset('csv', data_files="dataset/eval_final.csv")
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

# tokenized_train_dataset_word = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset_word = eval_dataset.map(tokenize_function, batched=True)
# print(tokenized_train_dataset_word["train"])



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


    
small_eval_dataset = tokenized_eval_dataset_word['train'].shuffle(seed=42).select(range(10))

model = BartForConditionalGeneration.from_pretrained("Bart_BPE_saved_models/checkpoint-390015")
# tokenizer = ByteLevelBPETokenizer("/content/drive/MyDrive/colab_backup/BPE_pretrained_on_code/")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./BPE_tokenizer/BPE.json")
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
                              "pad_token": "<pad>", "cls_token": "<s>"})
training_args = Seq2SeqTrainingArguments(output_dir = "./Bart_BPE_saved_models",\
                                         per_device_train_batch_size=2,\
                                         num_train_epochs=5,
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
        eval_dataset=small_eval_dataset ,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

predictions = trainer.predict(test_dataset=small_eval_dataset)
preds, labels,_ = predictions
if isinstance(preds, tuple):
    preds = preds[0]
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)


print(decoded_preds)