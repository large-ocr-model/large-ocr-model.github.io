import os
import sys
import random
from utils import is_valid


data_root = sys.argv[1]
output_path = sys.argv[2]
select_data_count = sys.argv[3]

filenames = os.listdir(data_root)
list_gallerys = []
for filename in filenames:
    file_path = os.path.join(data_root, filename)
    with open(file_path, 'r') as f:
        list_info = [l.strip() for l in f]
    for info in list_info:
        words = info.split(' ')
        clean_words = [word for word in words if is_valid(word)]
        list_gallerys +=clean_words


list_gallerys = random.sample(list_gallerys, select_data_count)

print("list_gallerys", len(list_gallerys))

open(output_path, 'w').write('\n'.join(list_gallerys)+'\n')
print("list_gallerys_clean", len(list_gallerys))