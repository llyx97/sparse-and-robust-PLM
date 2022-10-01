import os
from os.path import join

filenames = []
for root, dirs, files in os.walk('.'):
    for name in files:
        filename = join(root,name)
        if not filename.endswith('.sh'):
            continue

        file = open(filename, 'r')
        lines = file.readlines()
        file.close()

        for i, line in enumerate(lines):
            if 'export ROOT_DIR=' in line:
                lines[i] = 'export ROOT_DIR=$HOME/sparse-and-robust-PLM\n'

        file = open(filename, 'w')
        for line in lines:
            file.write(line)
        file.close()
