import os

directory = './data/english_recipe_flow_graph_corpus/r-200'

for FILE, DIR, ROOT in os.walk(directory):
    ROOT = sorted(ROOT)
    R_set = set([name.split('.')[0] for name in ROOT])
    for recipe in R_set:
        recipe_filename = os.path.join(FILE, recipe)
        text = ' '.join([el.split()[3] for el in open(recipe_filename+'.list', 'r', encoding='utf8').readlines()])
        with open(recipe_filename+'.txt', 'w', encoding='utf8') as write_file:
            write_file.write(text)
        
