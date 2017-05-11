import sys

assert len(sys.argv) == 2, 'in_file'
in_file = sys.argv[1]
out_file = in_file.replace('.csv', '.tab.txt')

def preprocess(line):
    chunks = line.strip().split(',')
    label = chunks[0]
    features = chunks[1:]
    new_chunks = [label] + features
    return '\t'.join(new_chunks)

with open(out_file, 'w') as outf:
    with open(in_file, 'r') as f:
        for line in f:
            outf.write(preprocess(line) + '\n')
