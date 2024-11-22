from typing import List, Dict

def flow2text(ner_list: List[str], flow_list: List[str]) -> str:
    '''
    This function converts nodes and edges
    from English Recipe Flowgraph Corpus (Yamakata 2020)
    into a single string.
    '''
    out_list = []
    ner: Dict = ner2dict(ner_list)
    flow: List[Dict] = flow2dictlist(flow_list)

    for line in flow:
        head = ner[line['head_triple_index']]
        edge = line['edge_tag']
        tail = ner[line['tail_triple_index']]
        out_list.append(f'[{head}, {edge}, {tail}]')
    out_text = ', '.join(out_list)
    return f'[{out_text}]'

def flow2dictlist(in_list: List[str]) -> List[Dict]:
    '''
    This function converts an NER list of strings
    from English Recipe Flowgraph Corpus (Yamakata 2020)
    into a list of dicts.
    '''
    out_list = []
    for line in in_list:
        temp_dict = {}
        line_list = line.split()
        temp_dict['head_triple_index'] = ' '.join(line_list[:3])
        temp_dict['edge_tag'] = line_list[3]
        temp_dict['tail_triple_index'] = ' '.join(line_list[4:])
        out_list.append(temp_dict)
    return out_list

def ner2dict(in_list: List[str]) -> Dict:
    '''
    This function converts an NER list of strings
    from English Recipe Flowgraph Corpus (Yamakata 2020)
    into a dict where each triple of indexes
    is a key of the dictionary.

    The indexes indicate the no. of the step, sentence, word
    '''

    out_dict = {}
    for line in in_list:
        line_list = line.split()
        id = ' '.join(line_list[:3])
        out_dict[id] = ' '.join(line_list[3:])
    return out_dict

def flow2ud(ner_list: List[str], flow_list: List[str], multi = False) -> Dict[str, List]:
    # I need to extract:
    # 'words' (FORM) -> words
    # 'pos_tags' (LABEL) -> ner tags
    # 'head_tags' (DEPREL) -> edge labels
    # 'head_indices' (HEAD) -> edge indices, adjacency matrix

    # for each line in ner_list look for the triple index of that line in the head indexes of flow_list
    # if you find it then HEAD is obtained by looking for the tail index of that line in flow_list
    # in ner_list

    # print('\n'.join(ner_list))
    # print('\n'.join(flow_list))
    # print('-------------------------')

    index_dict = {' '.join(line.split()[:3]): i + 1 for i, line in enumerate(ner_list)}

    data = {
        'words': [],
        'pos_tags': [],
        'head_tags': [],
        'head_indices': [],
        'step_indices': [],
        'sent_indices': [],
        'word_sent_indices': [],
    }

    for line in [el.strip() for el in ner_list]:
        line_list = line.split()
        triple_index = ' '.join(line_list[:3])
        
        flow_dicts = flow2dictlist(flow_list)
        HEAD_LIST = []
        DEPREL_LIST = []
        for entry in flow_dicts:
            if entry['head_triple_index'] == triple_index:
                HEAD_LIST.append(index_dict[entry['tail_triple_index']])
                DEPREL_LIST.append(entry['edge_tag'])
        if HEAD_LIST:
            assert HEAD_LIST[0] != 0
        HEAD_LIST = [0] if not HEAD_LIST else HEAD_LIST
        DEPREL_LIST = ['root'] if not DEPREL_LIST else DEPREL_LIST
        zipped_list = zip(HEAD_LIST[:1], DEPREL_LIST[:1])
        for head, deprel in zipped_list: # TODO: use all the edges and not just the first one
            data['words'].append(line_list[3])
            data['pos_tags'].append(line_list[-1])
            data['head_tags'].append(deprel)
            # raise NotImplementedError('is this supposed to be +1???')
            data['head_indices'].append(head)
            data['step_indices'].append(int(line_list[0]))
            data['sent_indices'].append(int(line_list[1]))
            data['word_sent_indices'].append(int(line_list[2]))

    return data

def main():
    ner = open('./data/yamakata/r-200/recipe-00000-05793.list')
    flow = open('./data/yamakata/r-200/recipe-00000-05793.flow')

    out = flow2text(ner, flow)
    print(out)

if __name__ == "__main__":
    main()