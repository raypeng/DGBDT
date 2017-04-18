import sys

assert len(sys.argv) == 2, 'in_file'
in_file = sys.argv[1]
out_file = in_file.replace('.txt', '.tab.txt')

def preprocess(line):
    chunks = line.strip().split(',')
    label = chunks[0]
    features = chunks[1:]
    new_chunks = features + [label]
    return '\t'.join(new_chunks)

with open(in_file, 'r') as f:
    lines = f.readlines()
with open(out_file, 'w') as f:
    f.write('\n'.join(preprocess(line) for line in lines))
