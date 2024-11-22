from utils import flow2text
import os
import re

DIR = './data/ERFC'

dataset = []

for ROOT, DIRS, FILES in os.walk(DIR):
    for file in FILES:
        if file.endswith('.flow'):
            entry = {}
            filepath = os.path.join(ROOT, file)
            filepath = re.match('(.*)\.flow', filepath).group(1)
            src = open(filepath + '.txt').read()
            ner = open(filepath + '.list').readlines()
            flow = open(filepath + '.flow').readlines()
            tgt = flow2text(ner, flow)
            entry['src'] = src
            entry['tgt'] = tgt
            dataset.append(entry)

import json

json_path = './data/ERFC/train.json'

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii = False)