import os
import sys
import time

file_name = sys.argv[1]
out_file_name = file_name.replace('.txt', '_clean.txt')
data_dir = os.path.dirname(file_name)

with open(file_name, 'r') as f:
    list_info = [l.strip() for l in f]

list_new_info = []

for info in list_info:
    img_path = info.split('\t')[0]
    img_path = os.path.join(data_dir, img_path)
    if os.path.isfile(img_path):
        list_new_info.append(info)
print(len(list_new_info))
open(out_file_name, 'w').write('\n'.join(list_new_info)+'\n')