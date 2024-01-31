
gallery_file_name=list_gallery.txt #
gallery_file_path=/path/to/your/$gallery_file_name
BG_IMG_PATH=/path/to/your/bg_img
FONT_PATH=/path/to/your/font

workspace=/path/to/your/generate_data_by_trgd
output_root=/path/to/your/output_data # 生成文件的位置
output_img_root=$output_root/gen_data/data
output_detail_gallery_root=$output_root/labels/details


# step1: gen detail gallery list
python gen_data_multi_thread.py \
--file_path ${gallery_file_path} \
--output_root ${output_detail_gallery_root} \
--font_root ${FONT_PATH} \
--background_rate 0.5 \
--bg_img_dir ${BG_IMG_PATH} \
--orientation_rate 0

# step2: gen data
cd TextRecognitionDataGeneratorss/trdg
input_file=$detail_gallery_file_root/$gallery_file_name
length=$(cat $input_file | wc -l)
echo $input_file $length
python run_new.py --output_dir $output_img_root/imgs \
-i $input_file \
-l cn \
-c $length \
-na 2 \
-w 1 \
-bl 1 \
-rbl \
-t 16 \
-tc "#000000,#282828"

# step3 postprocess for format
cd $output_img_root
LIST_FILES=$(find -name *.txt)

for file in ${LIST_FILES};
do

    SUB_DIR=${file::-10}
    echo $SUB_DIR


    SUB_DIR=${SUB_DIR//'.'/'\.'}
    SUB_DIR=${SUB_DIR//'/'/'\/'}
    echo $SUB_DIR

    sed -e "s/^/$SUB_DIR/g" $file > $file.temp
    sed -i "s/ /\t/g" $file.temp

done
LIST_FILES=$(find -name *.txt.temp)

root=$output_img_root
cat $LIST_FILES > $root/Label_data.txt
rm -rf *.txt.temp

# step4 check data for valid
cd $workspace
python check_data.py $root/Label_data.txt