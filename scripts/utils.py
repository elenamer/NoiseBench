from typing import Set

def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        if '\t' in line.strip():
            stripped_line = line.strip().split('\t')
        else:
            stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            sentence_tokens = [x[0] for x in point[:-1]]
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x

def save_to_column_file(filename, list):
    with open(filename, "w") as f:
        for sentence in list:
            for token in sentence:
                f.write('\t'.join(token))
                f.write('\n')
            f.write('\n')

