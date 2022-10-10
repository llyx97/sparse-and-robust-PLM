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
            if 'prune_after_ft' in line:
                lines[i] = lines[i].replace('prune_after_ft', 'mask_on_plm_std_ft')
            if 'prune_wo_ft' in line:
                lines[i] = lines[i].replace('prune_wo_ft', 'mask_on_plm_pt')

        file = open(filename, 'w')
        for line in lines:
            file.write(line)
        file.close()
