# script to transform attention pkl for visualize
#
# combined attn pkl is a dict of words in which each word is a dict with the following structure
# (each word)
# |
# +-- (layer)
# |   |
# |   +-- (encoder self-attention): tensor with size [(head), (beam), (target position), (source position)]
# |
# +-- (layer)
# |
# +-- ....
# |
# +-- (layer)
# |
# +-- ...
#
# this gets converted to dict of words in which each word is a dict with the following structure
# (word > layers > heads > segs):
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

    layers = {key: word[key] for key in word.keys() if key.startswith('layer') and key[5:].isdigit()}
    n_layer = len(layers)
    print("[INFO]     Number of layers: ", n_layer)
    init = True
    for layer_idx, layer in enumerate(layers):
        all_heads = layers[layer]['encoder_self_attention_weights']
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


def convert(checkpoint_n: int = 16):
    pkl_dir = os.environ["PKL_LOC"].split(',')
    pkl_path = os.path.join(os.getcwd(), f'{pkl_dir[0]} (checkpoint{checkpoint_n})', 'combined_attention.pkl')

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
    new_pkl_path = os.path.join(os.getcwd(), f'{pkl_dir[0]} (checkpoint{checkpoint_n})', 'combined_attention_aligned_for_analysis.pkl')
    with open(new_pkl_path, 'wb') as f:
        pickle.dump(res_dict, f)
    print("[INFO] pkl dumped")


if __name__ == "__main__":
    checkpoint_numbers = [16, 42, 49]
    for checkpoint_number in checkpoint_numbers:
        convert(checkpoint_number)

