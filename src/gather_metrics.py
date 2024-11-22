import json
import os
import pandas as pd

dir_path = './saved_models'
entry_list = []
cols = [
'bert_lhs_step_emb',
'parser_step_emb',
'format',
'bypass_w_lstm',
'seed',
'tagger_results',
'parser_labeled',
'parser_unlabeled',
'path',
]
for root, dirs, files in os.walk(dir_path):
    entry = {}
    if 'config.json' in files and 'test_results.json' in files:
        json_path = os.path.join(root, 'config.json')
        
        with open(json_path, 'r', encoding='utf8') as f:
            config_data = json.load(f)
        for el in cols[:-3]:
            if el not in config_data:
                config_data[el] = None
        entry.update(
            {
                cols[0]: config_data['bert_lhs_step_embeddings'],
                cols[1]: config_data['parser_step_embeddings_flag'],
                cols[2]: config_data['dataset_format'],
                cols[3]: config_data['bypass_w_lstm'],
                cols[4]: config_data['seed'],
            }
        )
        json_path = os.path.join(root, 'test_results.json')
        
        with open(json_path, 'r', encoding='utf8') as f:
            results_data = json.load(f)
        entry.update(
            {
                cols[5]: results_data['tagger_results']['F1'],
                cols[6]: results_data['parser_labeled_results']['F1'],
                cols[7]: results_data['parser_unlabeled_results']['F1'],
                cols[8]: root.split('_')[-1]
            }
        )

        entry_list.append(entry)

df = pd.DataFrame(entry_list, columns = cols)
df = df.sort_values(['format', 'seed'])
df = df.fillna(0)
df = df[df['format'] == 'ud']
df = df[df['bypass_w_lstm'] == 1]
print(df)
print(len(df))
df_grouped = df.groupby(['parser_step_emb'])

metric = 'parser_unlabeled'
print(f'Metric: {metric}')
print(df_grouped[metric].mean())
print(df_grouped[metric].std())
print()

metric = 'parser_labeled'
print(f'Metric: {metric}')
print(df_grouped[metric].mean())
print(df_grouped[metric].std())
print()

metric = 'tagger_results'
print(f'Metric: {metric}')
print(df_grouped[metric].mean())
print(df_grouped[metric].std())
print()