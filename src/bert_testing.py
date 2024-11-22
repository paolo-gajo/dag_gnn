from transformers import BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertModel

# Load model directly
from transformers import RoFormerTokenizer, RoFormerModel, RoFormerConfig

config = RoFormerConfig.from_pretrained("junnyu/roformer_chinese_small")

tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_small")
model = RoFormerModel(config)

# config = BertConfig.from_pretrained("bert-base-cased")

# tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = BertModel(config)

input = tokenizer('test sentence here', return_tensors = 'pt')

output = model(**input)

print(output)