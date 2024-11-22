from src.yamakata_utils.utils import flow2ud
import os
import re
from sklearn.model_selection import train_test_split

DIR = './data/yamakata'

data = []
all_head_tags = []
pos_tags = []

for ROOT, DIRS, FILES in os.walk(DIR):
    for file in FILES:
        if file.endswith('.flow'):
            entry = {}
            filepath = os.path.join(ROOT, file)
            filepath = re.match('(.*)\.flow', filepath).group(1)
            ner = open(filepath + '.list').readlines()
            flow = open(filepath + '.flow').readlines()
            entry = flow2ud(ner, flow)
            data.append(entry)
            all_head_tags.extend(entry['head_tags'])
            pos_tags.extend(entry['pos_tags'])

print(len(set(all_head_tags)))
print(len(set(pos_tags)))

import json
print(filepath)

json_path = './data/yamakata/efrc_ud.json'

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii = False)

train, test = train_test_split(data, test_size=0.2, random_state=42)
print(len(train))

val, test = train_test_split(test, test_size=0.5, random_state=42)
print(len(val))
print(len(test))

json_path_train = json_path.replace('.json', '_train.json')

with open(json_path_train, 'w', encoding='utf8') as f:
    json.dump(train, f, ensure_ascii = False)

json_path_val = json_path.replace('.json', '_val.json')

with open(json_path_val, 'w', encoding='utf8') as f:
    json.dump(val, f, ensure_ascii = False)

json_path_test = json_path.replace('.json', '_test.json')

with open(json_path_test, 'w', encoding='utf8') as f:
    json.dump(test, f, ensure_ascii = False)

