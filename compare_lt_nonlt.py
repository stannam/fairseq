import torch
import pickle as pk
import os


class Grid:
    def __init__(self, nrow:int=4, ncol:int=2):
        # nrow: number of rows. rows represent layers
        # ncol: number of columns. columns represent heads
        #
        # (grid: list of 4 layers)
        # |
        # +-- (layer: list of 2 heads)
        # |   |
        # |   +-- (head: dict containing attn information for that specific layer-head combo)
        # |   |
        # |   +-- (head: dict containing attn information for that specific layer-head combo)

        self.nrow = nrow
        self.ncol = ncol
        self.grid = [[{} for _ in range(ncol)] for _ in range(nrow)]

    def get(self, row: int, col: int) -> dict:
        return self.grid[row][col]

    def merge_dicts(self, dict1, dict2) -> dict:
        if len(dict1.keys()) == 0:
            # dict1 is empty.
            return dict2

        merged_dict = {}

        for k in dict1.keys():
            # dict2[k].dim() is always 0
            if dict1[k].dim() == 0 and dict2[k].dim() == 0:
                merged_dict[k] = torch.stack([dict1[k], dict2[k]])
            else:
                if dict2[k].dim() == 0:
                    merged_dict[k] = torch.cat((dict1[k], dict2[k].unsqueeze(0)))
                else:
                    merged_dict[k] = torch.cat((dict1[k], dict2[k]))
        return merged_dict

    def set(self, row: int, col: int, value: dict) -> None:
        self.grid[row][col] = value

    def merge(self, other: 'Grid') -> 'Grid':
        self.check_dimension(other)
        if self.nrow != other.nrow or self.ncol != other.ncol:
            raise ValueError("Grid dimensions must match for merging.")

        if 'Empty' in str(self):
            # empty grid
            return other

        merged_grid = Grid(self.nrow, self.ncol)
        for i in range(self.nrow):
            for j in range(self.ncol):
                merge_dict = self.merge_dicts(self.get(i, j), other.get(i, j))
                merged_grid.set(i, j, merge_dict)

        return merged_grid

    def compare(self, other: 'Grid'):
        return 0

    def check_dimension(self, other: 'Grid'):
        if self.nrow != other.nrow or self.ncol != other.ncol:
            raise ValueError("Grid dimensions must match for merging.")

    def __str__(self):
        if self.grid[0][0] == {}:
            return "Empty Grid"
        try:
            n_items = len(self.grid[0][0]['word_initial'])
        except TypeError:
            n_items = 1
        return f"Grid of size {len(self.grid)}x{len(self.grid[0])} with {n_items} items"


def process_one_matrix(syllabified: list, attns, target_idx: int):
    # process a single head-layer
    wi = attns[0]
    wf = attns[-1]
    preceeding = attns[target_idx - 1]

    i = 0
    syll_attn = []
    vowel_attn = []
    for syl in syllabified:
        syl_sum = attns[i:i + len(syl)].sum()
        v_attn = 0
        for syl_i, segment in enumerate(syl):
            if segment == "V":
                v_attn += attns[i+syl_i]
        i += len(syl)
        syll_attn.append(syl_sum)
        vowel_attn.append(v_attn)
    return wi, wf, preceeding, syll_attn, vowel_attn


def process_one_item(ex: dict, stim_dict: dict, analysis_type: str):
    # ex: example. result of translation with attention weights info
    # stim_dict: stimuli dictionary with 'transcription' as the key and other info as value
    syl_code_table = {
        # for parsing stimulus code
        "1": "CVC",
        "2": "CV",
        "3": "VC",
        "4": "V"
    }

    stim = stim_dict.get(ex['input_word'], None)
    layers = ex['layers']

    # parse code and get syllable structure
    code = stim['code']  # 'A-3-1-1'
    code = code.split('-')
    sti_class = code[0]

    syllabified = [syl_code_table.get(code[i]) for i in [1, 2, 3]]

    input_w = ex['input_word'].split(' ')

    if sti_class in ['A', 'B']:
        # manipulation in the first segment. first syllable has additional onset
        if len(code) == 5:
            syllabified[0] = 'CVC'

    if sti_class in ['A', 'C']:
        # LT between second and third syllables -> tensifiable = onset of syl3
        tensifiable_idx = len(syllabified[0]) + len(syllabified[1])
    else:
        # LT between first and second syllables -> tensifiable = onset of syl2
        tensifiable_idx = len(syllabified[0])

    # if input includes a diphthong, need to modify syllable structure representation
    if sum(len(syl) for syl in syllabified) != len(input_w):
        i = 0
        for idx, syl in enumerate(syllabified):
            this_syllable = input_w[i:i + len(syl)]
            if any(seg in ['w', 'y'] for seg in this_syllable):
                # this syllable has a glide!
                if syl[0] == 'V':
                    syl = 'V' + syl
                else:
                    syl = syl[0] + 'V' + syl[1:]
                syllabified[idx] = syl
            i += len(syl)

    # if diphthong appears before the LT location, tensifiable_idx += 1
    for i in range(tensifiable_idx):
        if input_w[i] in ['w', 'y']:
            tensifiable_idx += 1

    # process all layers and heads and create grid
    # e.g., result_grid[3][1] => layer #3, head #1
    result_grid = Grid()
    # result_grid = [
    #     [[], []],  # layer 0
    #     [[], []],  # layer 1
    #     [[], []],
    #     [[], []],
    # ]

    for i in range(len(layers)):  # number of layers
        for j in range(len(layers[i])):  # number of heads
            wi, wf, preceeding, syll_attn, vowel_attn = process_one_matrix(syllabified,
                                                                           layers[i][j][tensifiable_idx],
                                                                           tensifiable_idx)
            one_cell = {'word_initial': wi,
                        'word_final': wf,
                        'preceeding': preceeding,
                        'syl0_sum': syll_attn[0],
                        'syl0_vowel': vowel_attn[0],
                        'syl1_sum': syll_attn[1],
                        'syl1_vowel': vowel_attn[1],
                        'syl2_sum': syll_attn[2],
                        'syl2_vowel': vowel_attn[2]
                        }
            result_grid.grid[i][j] = one_cell

    if analysis_type == 'stimuli':
        # L-Tensified?
        LT = not (ex['input_word'] == ex['output_word'])  # LT True if tensified
        return sti_class, LT, result_grid

    else:
        return sti_class, result_grid


def process_filler(pkld_data, transcription_index):
    master_result_dict = {'12_initial': Grid(),
                          '12_medial': Grid(),
                          '23_initial': Grid(),
                          '23_medial': Grid(),
                          }
    for item_n, token in enumerate(pkld_data):
        print(f'Processing #{item_n}')
        sti_class, grid = process_one_item(pkld_data[token], stim_dict=transcription_index, analysis_type='filler')

        pigeonhole = ''
        if sti_class == 'A':
            pigeonhole += '23_initial'
        elif sti_class == 'B':
            pigeonhole += '12_initial'
        elif sti_class == 'C':
            pigeonhole += '23_medial'
        elif sti_class == 'D':
            pigeonhole += '12_medial'

        master_result_dict[pigeonhole] = master_result_dict[pigeonhole].merge(grid)
    print(master_result_dict)

    return master_result_dict


def process_stimuli(pkld_data, transcription_index):
    master_result_dict = {'lt_12_initial': Grid(),
                          'lt_12_medial': Grid(),
                          'lt_23_initial': Grid(),
                          'lt_23_medial': Grid(),
                          'nonlt_12_initial': Grid(),
                          'nonlt_12_medial': Grid(),
                          'nonlt_23_initial': Grid(),
                          'nonlt_23_medial': Grid()}

    for item_n, token in enumerate(pkld_data):
        print(f'Processing #{item_n}')
        sti_class, LT, grid = process_one_item(pkld_data[token], stim_dict=transcription_index, analysis_type='stimuli')

        pigeonhole = ''
        if LT:
            pigeonhole += 'lt_'
        else:
            pigeonhole += 'nonlt_'

        if sti_class == 'A':
            pigeonhole += '23_initial'
        elif sti_class == 'B':
            pigeonhole += '12_initial'
        elif sti_class == 'C':
            pigeonhole += '23_medial'
        elif sti_class == 'D':
            pigeonhole += '12_medial'

        master_result_dict[pigeonhole] = master_result_dict[pigeonhole].merge(grid)
    print(master_result_dict)

    return master_result_dict


def main(pickle_base_path: str, reference_table_path: str, analysis_type: str = None) -> None:
    if analysis_type is None or analysis_type not in ['stimuli', 'filler']:
        print('Analysis type is not specified or specified incorrectly.')
        return

    pickle_path = os.path.join(pickle_base_path, 'combined_attention_aligned_for_analysis.pkl')

    pkld_data = pk.load(open(pickle_path, 'rb'))  # type(pkld_data) = dict

    wordtokens = list(pkld_data.keys())
    inputs = [pkld_data[token]['input_word'] for token in wordtokens]
    outputs = [pkld_data[token]['output_word'] for token in wordtokens]

    stimuli = []

    with open(reference_table_path, 'r', encoding='utf-8') as file:
        headers = file.readline().strip().split('\t')
        for line in file:
            values = line.strip().split('\t')
            entry = dict(zip(headers, values))
            stimuli.append(entry)

    transcription_index = {entry['transcription']: entry for entry in stimuli}

    if analysis_type == 'stimuli':
        result_dict = process_stimuli(pkld_data, transcription_index)
    else:
        result_dict = process_filler(pkld_data, transcription_index)

    # save as pickle
    pickle_export_path = os.path.join(pickle_base_path, 'compare_lt_nonlt_attn.pkl')
    with open(pickle_export_path, 'wb') as f:
        pk.dump(result_dict, f)


if __name__ == '__main__':
    pkl_base_path = 'E:\\Dropbox\\Dissertation\\2024-Jan Fairseq\\segment no syll boundary\\2024-04-25 4layer_2head'
    ref_base_path = "E:\\Dropbox\\Dissertation\\Korean experiment\\stimuli"

    # stimuli (LT vs non LT)
    pickle_path = os.path.join(pkl_base_path, 'export_attention_weights')
    reference_table_path = os.path.join(ref_base_path, "2024_06_07_stimuli_regrouped_in_Park's_convention.tsv")
    main(pickle_path, reference_table_path, analysis_type='stimuli')

    # fillers (no LT environment)
    #pickle_path = os.path.join(pkl_base_path, 'export_attention_weights_filler')
    #reference_table_path = os.path.join(ref_base_path, "2024_06_11_fillers_in_Park's_convention.tsv")
    #main(pickle_path, reference_table_path, analysis_type='filler')
