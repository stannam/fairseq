import os
import pickle
from datetime import datetime

from fairseq.models.transformer import TransformerModel
from fairseq.data import Dictionary

# make sure to set '2024-01-26 model_outputs' as the working directory
# and also environment variable for pickle

CWD = os.getcwd()
DATA_BIN = os.path.join(CWD, 'bin')
DICTIONARY = os.path.join(DATA_BIN, 'dict.ur.txt')

# if visualizing the 'functional' transformer-tiny model
MODEL = os.path.join(CWD, 'model_output_transformer')
CHECKPOINT = 'checkpoint97.pt'

# visualizing the 'complex' transformer model
#MODEL = os.path.join(CWD, '2024-03-14 transformer_output')
#CHECKPOINT = 'checkpoint3.pt'

# visualizing an example from fairseq
#MODEL = os.path.join(CWD, 'wmt14.en-fr.joined-dict.transformer')  # a fairseq example model
#DATA_BIN = os.path.join(CWD, 'wmt14.en-fr.joined-dict.newstest2014')
#DICTIONARY = os.path.join(DATA_BIN, 'dict.en.txt')
#CHECKPOINT = 'model.pt'


def pkl_handler(dest_path: str,
                key_to_write: str = None,
                value_to_write: any = None,
                new: bool = False,
                flatten_key: tuple = 0) -> None:
    if key_to_write is None:
        # create new pkl file.
        dummy = {}
        with open(dest_path, 'wb') as file:
            pickle.dump(dummy, file)
        return

    with open(dest_path, 'rb') as file:
        # Load the dummy pickled object for the first time
        attention_pickle = pickle.load(file)

    if new:
        attention_pickle[flatten_key] = {'input_word': value_to_write,
                                         'finished': False}
    else:
        attention_pickle[flatten_key][key_to_write] = value_to_write
        attention_pickle[flatten_key]['finished'] = True

    with open(dest_path, 'wb') as file:
        # Pickle the 'data' dictionary and write it to the file
        pickle.dump(attention_pickle, file)
    return


def extract_attention(model, input_word: str, pkl_path:str):
    # overriding input_word for debugging
    input_word = "t v k m a ng"  # output: t v ngZ m a ngZ
    print(f'Input: {input_word}')
    src_tokens = model.encode(input_word)
    flatten_key = tuple(src_tokens.numpy().flatten())

    # add input_word to pickle
    pkl_handler(dest_path=pkl_path, new=True, key_to_write='input_word',
                value_to_write=input_word, flatten_key=flatten_key)

    output_word = model.translate(sentences=input_word, verbose=True)
    print(f'Output: {output_word}')

    # add output_word to pickle
    pkl_handler(dest_path=pkl_path, key_to_write='output_word', value_to_write=output_word, flatten_key=flatten_key)


def main():
    # Initial printout to help myself...
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[INFO] The start time is {now}.')
    pkl_dir = os.environ["PKL_LOC"].split(',')
    pkl_path = os.path.join(os.getcwd(), pkl_dir[0], f'{pkl_dir[1]}.pkl')
    print(f'[INFO] Working directory: {os.getcwd()}\n[INFO] Pickle location: {pkl_path}')

    # Script actually starts here.

    # import the pretrained transformer model and dictionary
    model = TransformerModel.from_pretrained(MODEL, checkpoint_file=CHECKPOINT, data_name_or_path=DATA_BIN)
    underlying_model = model.models[0]  # since 'model' is a GeneratorHubInterface instance, wrapping the actual model.
    dictionary = Dictionary.load(DICTIONARY)

    # prepare pkl file to save attention weights.
    pkl_handler(pkl_path)

    # load the wordlist I used for the human experiment
    experiment_wordlist_path = os.path.join('2024-03-08 apply_translation', 'entries.txt')
    with open(experiment_wordlist_path, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]

    # iterate over the words
    for i, word in enumerate(words):
        print(f'[DEBUG] start working on word #{i}: {word}.')

        extract_attention(model=model,
                          input_word=word,
                          pkl_path=pkl_path
                          )


if __name__ == '__main__':
    main()