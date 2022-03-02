from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from datasets import load_dataset, load_metric,concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
from tokenizers.processors import TemplateProcessing
import os

path = "eval"
methods = load_dataset('text', data_files= path  + "_final.methods")
tests = load_dataset('text', data_files= path  + "_final.tests")
paths = ["eval_final.methods","eval_final.tests"]
assert methods['train'].features.type == tests["train"].features.type
concat_data = concatenate_datasets([methods['train'],tests['train']])
print(concat_data)
def batch_iterator(batch_size=1000):
    for i in range(0, len(concat_data), batch_size):
        yield concat_data[i : i + batch_size]["text"]

WordPiece_tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
WordPiece_tokenizer.normalizer = normalizers.NFKC()
WordPiece_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
WordPiece_tokenizer.decoder = decoders.WordPiece()


trainer = trainers.WordPieceTrainer(
    vocab_size=50000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<s>",
                  "<pad>",
                  "</s>",
                  "<unk>",
                  "<mask>",]
)
WordPiece_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[("<s>", 0), ("</s>", 2)],
)
WordPiece_tokenizer.enable_padding(pad_id=1,pad_token="<pad>")
WordPiece_tokenizer.enable_truncation(max_length=512)

WordPiece_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

directory = "WordPiece_tokenizer"
if os.path.exists(directory)==False:
  os.mkdir(directory)
WordPiece_tokenizer.save("WordPiece_tokenizer/WordPiece.json")
