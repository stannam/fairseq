"""
parse translation results and get L-Tensification score
translation results look like e.g.,
2024-07-13 13:50:30 | INFO | fairseq.hub_utils | S	i l t a m x n
2024-07-13 13:50:30 | INFO | fairseq.hub_utils | H	-0.04657294228672981	i l tt a m x n
2024-07-13 13:50:30 | INFO | fairseq.hub_utils | P	-0.0000 -0.0000 -0.0464 -0.0000 -0.0000 -0.0000 -0.0000 -0.0001
2024-07-13 13:50:30 | INFO | fairseq.hub_utils | H	-3.0945215225219727	i l t a m x n
2024-07-13 13:50:30 | INFO | fairseq.hub_utils | P	-0.0000 -0.0000 -3.0944 -0.0000 -0.0000 -0.0000 -0.0000 -0.0001

"""
import os
import math


def main():
    log_path = os.path.join(os.getcwd(),'translation_log.txt')
    need_quit = input(f'log path: {log_path} \neixsts? {os.path.exists(log_path)}\nQ to exit? ')
    if need_quit.lower() == 'q':
        return

    # read raw text
    with open(log_path, 'r', encoding='utf-8') as f:
        raw_txt = [line.strip() for line in f.readlines()]

    r_txt = parse(raw_txt)

    return_path = os.path.join(os.getcwd(),'translation_log_parsed.txt')
    with open(return_path, 'w') as file:
        file.write(r_txt)


def parse(input_list: list) -> str:
    # parse raw text
    idx = 0
    output_str = ''
    while True:
        this_line = input_list[idx].split('\t')
        if this_line[0] == 'S':
            H1 = input_list[idx + 1].split('\t')
            H2 = input_list[idx + 3].split('\t')

            H1_score, H2_score = get_scores(H1, H2)
            score = H1_score if len(H1[2]) > len(H2[2]) else H2_score
            conf = input(f"\n{this_line}\n{H1_score}\t{H1[2]}\n{H2_score}\t{H2[2]}\nLT score\t{score}\nLook ok? (y/n) ")
            if conf.lower() == 'n':
                print("do it yourself!")
                for i in range(11):
                    print(input_list[idx + i])
                score = float(input("score?"))

            output_str += f'{this_line[1]}\t{score}\n'
            idx += 11
        if idx >= len(input_list):
            break

    return output_str


def get_scores(H1: list, H2: list) -> tuple:
    # get two lists, return two lists
    H1_raw_score = float(H1[1])
    H2_raw_score = float(H2[1])
    return math.exp(H1_raw_score), math.exp(H2_raw_score)

if __name__ == "__main__":
    main()
