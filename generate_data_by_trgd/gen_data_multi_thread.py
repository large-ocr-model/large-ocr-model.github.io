import os
import glob
import time
import random
import argparse
from collections import Counter
from collections import defaultdict as ddt
from threading import Thread,Lock

from utils import check_font_chars, load_font, is_number, is_alphabet, check_font_chars, is_number_or_eng



def get_support_fonts(fonts, info):
    list_support_fonts = []
    for k, v in fonts.items():
        try:
            unsupported_chars, _ = check_font_chars(v, info)
            if len(unsupported_chars) ==0:
                list_support_fonts.append(k)
        except:
            print("Warning:", k)
    return list_support_fonts

def get_chinese_font(fonts, info, FONTS_ALL):
    font = random.choice(fonts)
    unsupported_chars = not_support_fonts[font]
    unsupported_sub_chars = [char for char in info if char in set(unsupported_chars)]
    idx = 0
    while len(unsupported_sub_chars) != 0:
        font = random.choice(fonts)
        unsupported_chars = not_support_fonts[font]
        unsupported_sub_chars = [char for char in info if char in set(unsupported_chars)]
        idx+=1
        if idx >=5:
            list_support_fonts = get_support_fonts(FONTS_ALL, info)
            if not len(list_support_fonts):
                return None, unsupported_sub_chars
            else:
                font = random.choice(list_support_fonts)
                return font, []
    return font, []

def gen_label_detail(process_id, list_info, fonts, not_support_fonts, list_all_support_fonts, list_num_fonts, colors, num_fonts_rate, image_dir, orientation_rate=0, distorsion_rate=0, skew_rate=0, background_rate=1, output_file=None):
    global list_info_new_all, list_vertical_new_all, list_no_font_all, FONTS_ALL
    list_info_new = []
    list_vertical_new = []
    random.seed(0)
    st = time.time()

    count = 0
    list_no_font = []
    with open(output_file+f'.{process_id}', 'w') as w_f:
        for info in list_info:
            # 字体
            if is_number_or_eng(info) and random.random() <num_fonts_rate:
                font = random.choice(list_num_fonts)
            else:
                font, unsupported_chars = get_chinese_font(fonts, info, FONTS_ALL)
            if not font:
                list_no_font.append(info+'\t'+' '.join(unsupported_chars))
                continue
            # 颜色
            color = random.choice(colors)
            # 是否竖向

            orientation = 1 if random.random() < orientation_rate else 0
            if len(info) > 25:
                orientation = 0
            # 是否弯曲
            distorsion = random.choice([1, 2]) if random.random() < distorsion_rate else 0
            distorsion_orientation = 2

            # 是否倾斜
            skew_angle = 5 if random.random() <skew_rate else 0
            random_skew = True

            # 添加背景
            background = 3 if random.random() <background_rate else 1
  

            info_detail = [info, font, color, orientation, distorsion, distorsion_orientation, skew_angle, random_skew, background, image_dir]
            info_detail = [str(info) for info in info_detail]
            info_detail = '\t'.join(info_detail)


            if orientation == 1:
                list_vertical_new.append(info_detail)
            else:
                list_info_new.append(info_detail)
                w_f.write(info_detail+'\n')

            count +=1
            if count %1000 == 0:
                print("cost time:{} process: {}".format(time.time() - st, round(count/len(list_info), 4)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen label gallery')
    parser.add_argument('--output_root', default='', type=str, help='output path')
    parser.add_argument('--font_root', default='FONTS', type=str)
    parser.add_argument('--colors', default=['#9e5526', '#000000', '#1d307c', '#3a2e20', '#8c775a', '#657868', '#7a2d28', 
                                             '#5a4645', '#434232', '#433230', '#3e5180', '#262f84', '#9a645b', '#7c3024', 
                                             '#2c2c37', '#647b68', '#021d10', '#292929', '#4e4e4e', '#777777', '#a2a2a2', 
                                             '#d0d0d0', '#000000,#a2a2a2'], type=str)
    parser.add_argument('--only_latttice', action="store_true")
    parser.add_argument('--file_path', default='', type=str)
    parser.add_argument('--bg_img_dir', default = '', type=str)
    parser.add_argument('--num_fonts_rate', default = 0.2, type=float)
    parser.add_argument('--orientation_rate', default = 0.2, type=float)
    parser.add_argument('--distorsion_rate', default = 0.2, type=float)
    parser.add_argument('--skew_rate', default = 0.2, type=float)
    parser.add_argument('--background_rate', default = 0.2, type=float)
    args = parser.parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    file_path = args.file_path
    output_file = os.path.join(args.output_root, os.path.basename(file_path))
    
    with open(file_path, 'r') as f:
        list_info = [l.strip() for l in f]

    list_info = list_info #* 1000


    cur_fonts = []
    list_num_fonts = []
    for root, dirs, files in os.walk(args.font_root):
        for file in files:
            if file.endswith("ttf") or file.endswith("TTF") or file.endswith("ttc"):
                file_path = os.path.join(root, file)
                cur_fonts.append(file_path)
                if 'lattice/number' in file_path:
                    list_num_fonts.append(file_path)
    print(len(cur_fonts))
    FONTS_ALL = {}
    for p in cur_fonts:
        ttf = load_font(p)
        FONTS_ALL[p] = ttf


    list_support_fonts = []

    for k, v in FONTS_ALL.items():
        try:
            unsupported_chars, _ = check_font_chars(v, '我')
            if len(unsupported_chars) ==0:
                list_support_fonts.append(k)
        except:
            print("Warning:", k)

    charset = Counter(''.join(list_info))
    not_support_fonts = {}

    list_all_support_fonts = []
    for k in list_support_fonts:
        try:
            unsupported_chars, _ = check_font_chars(FONTS_ALL[k], charset)
            not_support_fonts[k] = set(unsupported_chars)
            if len(unsupported_chars) == 0:
                list_all_support_fonts.append(k)
            # print(k, "unsupported_chars", len(unsupported_chars))
        except:
            print("Warning:", k)
    print("list_all_support_fonts", len(list_all_support_fonts))

    print("list_support_fonts", len(list_support_fonts), len(not_support_fonts))

    # start gan data
    lock=Lock()
    l=[]
    partition = 16

    step = len(list_info)//partition
    list_info_new_all, list_vertical_new_all, list_no_font_all = [], [], []
    st = time.time()
    for i in range(partition):
        p=Thread(target=gen_label_detail, args=(i, list_info[i*step:min((i+1)*step, len(list_info))],list_support_fonts, not_support_fonts, list_all_support_fonts, list_num_fonts, args.colors, 
                        args.num_fonts_rate, 
                        args.bg_img_dir,
                        args.orientation_rate, 
                        args.distorsion_rate, 
                        args.skew_rate, 
                        args.background_rate, output_file))
        l.append(p)
        p.start()
    for p in l:
        p.join()


    cmd = f'cat {output_file}.* > {output_file}; rm -rf {output_file}.*'
    os.system(cmd)
