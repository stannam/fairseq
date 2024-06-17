# extract_attentions.py returns many pkl for practical reasons
# this script combines all attention pickles
import os
import pickle

def process_pickle(data: dict):
    new_data = {
        'input_word': data['input_word'],
        'output_word': data['output_word'],


    }
    idx = data['winner_idx']


def combine():
    pkl_dir = os.environ["PKL_LOC"].split(',')
    pkl_path = os.path.join(os.getcwd(), pkl_dir[0])

    print(f'pickles directory: {pkl_path}')
    need_quit = input("\n Make sure the info above makes sense. Q to quit.")
    if need_quit.lower() == 'q':
        return
    pickle_list = [file for file in os.listdir(pkl_path) if file.startswith("attentions_") and file.endswith(".pkl")]

    print(f'[INFO] number of pickle files: {len(pickle_list)}')

    combined = {}

    for file in pickle_list:
        print(f'[INFO] processing file: {file}')
        file_path = os.path.join(pkl_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        #process_pickle(data)
        combined.update(data)

    output_path = os.path.join(pkl_path, 'combined_attention.pkl')
    with open(output_path, 'wb') as output_file:
        pickle.dump(combined, output_file)


if __name__ == "__main__":
    combine()
