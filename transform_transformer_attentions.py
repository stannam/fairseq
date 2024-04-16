# script to transform attention pkl for visualize
#
# combined attn pkl is a dict of words in which each word is a dict with the following structure
# (each word)
# |
# +-- (segment)
# |   |
# |   +-- (layer)
# |   |   |
# |   |   +-- (decoder self-attention)
# |   |   |
# |   |   +-- (cross-attention): tensor with size [(head), (beam), (input_seg)]
# |   |
# |   +-- (layer)
# |   |
# |   +-- ....
# |
# +-- (segment)
# |
# +-- ...
#
# this gets converted to dict of words in which each word is a dict with the following structure:
# (each word)
# |
# +-- (layers: list of layers)
# |   |
# |   +-- (layer: list of attention heads)
# |   |   |
# |   |   +-- (attention head: list of n-many to-input-attn Tensors where n=output_length)
# |   |   |
# |   |   +-- ...
# |   |
# |   +-- (layer)
# |   |
# |   +-- ....
# |
# +-- (layers)
# |
# +-- ...
# |
# +-- input_word: str
# |
# +-- output_word: str
#
# and then dumped as 'combined_attention_aligned_for_analysis.pkl'


import pickle
import os


def parse_word(word: dict) -> dict:
    winner_idx = word.get('winner_idx')
    r_dict = {      # container for outermost dictionary
        'input_word': word.get('input_word'),
        'output_word': word.get('output_word'),
        'layers': list()}

    segments = {key: word[key] for key in word if key.startswith('seg') and key[3:].isdigit()}
    print("[INFO] Number of seg: ", len(segments))
    init = False
    for seg_idx, seg in enumerate(segments):
        print("[INFO]     Segment #", seg_idx)
        if seg_idx == 0:
            init = True
        layers = {key: segments[seg][key] for key in segments[seg].keys() if key.startswith('layer') and key[5:].isdigit()}
        n_layer = len(layers)
        print("[INFO]     Number of layers: ", n_layer)
        for layer_idx, layer in enumerate(layers):
            all_heads = layers[layer]['cross_attention_weights']
            n_heads = len(all_heads)
            print("[INFO]         In layer #", layer_idx)
            print("[INFO]         Number of heads: ", n_heads)
            if init:
                [r_dict['layers'].append(list()) for i in range(n_layer)]
                for each_layer in range(n_layer):
                    [r_dict['layers'][each_layer].append(list()) for head in range(n_heads)]
                init = False
            for head_idx, head in enumerate(all_heads):
                tensor_to_add = head[winner_idx]
                r_dict['layers'][layer_idx][head_idx].extend(tensor_to_add)
    return r_dict

def convert():
    pkl_dir = os.environ["PKL_LOC"].split(',')
    pkl_path = os.path.join(os.getcwd(), pkl_dir[0], 'combined_attention.pkl')

    print("[INFO] cwd: ", os.getcwd())
    print("[INFO] pkl_path (path to combined pkl_attn: ", pkl_path)
    need_quit = input("Ok to proceed? Q to quit.")
    if need_quit.lower() == 'q':
        return

    # load pkl file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print("[INFO] pkl loaded")

    res_dict = {}

    for idx, word in enumerate(data):
        print(f'[INFO] {idx}th word: {data[word]["input_word"]}')
        one_word_cross_attn_dict = parse_word(data[word])
        res_dict[word] = one_word_cross_attn_dict
        print('\n')

    # save pkl file
    new_pkl_path = os.path.join(os.getcwd(), pkl_dir[0], 'combined_attention_aligned_for_analysis.pkl')
    with open(new_pkl_path, 'wb') as f:
        pickle.dump(res_dict, f)
    print("[INFO] pkl dumped")


if __name__ == "__main__":
    convert()
