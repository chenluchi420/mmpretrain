from mmpretrain import list_models
from mmpretrain import get_model
from mmpretrain import ImageClassificationInferencer
import os 
from tqdm import tqdm
import cv2
import numpy as np

def crop_image(bnb_point, input_image):
    #X_center, Y_center, X_width, Y_height
    input_image_height, input_image_width = input_image.shape[0], input_image.shape[1]
    X_center, Y_center = int(bnb_point[0]*input_image_width), int(bnb_point[1]*input_image_height)
    X_width, Y_height = int(bnb_point[2]*input_image_width), int(bnb_point[3]*input_image_height)

    left_top_x, left_top_y =int(X_center - (X_width/2)), int(Y_center - (Y_height/2))
    # print(left_top_x)
    crop_image = input_image[left_top_y: left_top_y+ Y_height, left_top_x: left_top_x+ X_width] 
    return crop_image

inferencer = ImageClassificationInferencer('/home/chenzy/mmpretrain/efficientnetv2-b0_sea_trash.py', pretrained='/home/chenzy/mmpretrain/work_dirs/efficientnetv2-b0_sea_trash/epoch_100.pth')

input_img_path = '/home/chenzy/yolov7_sea/runs/detect/exp3'
input_label_path =  '/home/chenzy/yolov7_sea/runs/detect/exp3/labels'
lable_file_list = os.listdir(path=input_label_path)
image_file_list = os.listdir(path=input_img_path)
output_lable = '/home/chenzy/mmpretrain/no_trash/new_lable'


for i in range(len(image_file_list)):
    image_file_list[i] = str(input_img_path)+ "/"+ image_file_list[i]

list_name = ['WASTE_1', 'WASTE_2', 'WASTE_3', 'WASTE_4', 'WASTE_5', 'WASTE_6', 'WASTE_7', 'WASTE_8', 'WASTE_9', 'WASTE_10', \
         'WASTE_11', 'WASTE_12', 'WASTE_13', 'WASTE_14', 'WASTE_15', 'WASTE_16', 'WASTE_17', 'WASTE_18', 'WASTE_19', 'WASTE_20']

for i in tqdm(range(len(lable_file_list))):
    # try:
    print(lable_file_list[i])
    label_file_path = input_label_path+ "/"+ lable_file_list[i]
    image_file_path = input_img_path+ "/"+ lable_file_list[i].replace(".txt","")+ ".jpg"
    if image_file_path not in image_file_list:
        image_file_path = image_file_path.replace(".jpg",".JPG")
    
    label_file =  open(label_file_path,'r')
    for line in label_file.readlines()[:-1]:
        
        line = line.split(" ")
        bnb_point = float(line[1]), float(line[2]), float(line[3]), float(line[4].strip())
        
        data_path = open(image_file_path,"rb")
        bytes = bytearray(data_path.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        img = crop_image(bnb_point, image)
        result = inferencer(img)[0]
        # print(result['pred_class'])
        # print(result['pred_score'])
        
        '''
        str_obj = result['pred_class']
        list_name.index(str_obj)
        line[0] = list_name.index(str_obj)
        line[-1] = str(result['pred_score'])

        new_line = str(line[0])+ " "+ str(line[1])+ " "+ str(line[2])+ " "+ str(line[3])+ " "+str(line[4])+ " "+ str(line[5])+ "\n"
        new_lable_txt_file = output_lable + '/'+ lable_file_list[i]
        new_label_file =  open(new_lable_txt_file,'a')
        new_label_file.write(new_line)'''
        
# /home/chenzy/mmpretrain/work_dirs/efficientnetv2-b0_sea_trash/last_checkpoint
    


# image_file= '/home/chenzy/mmpretrain/no_trash/images/20230914_海廢初賽拍攝(101).JPG'
# results = inferencer(image_file)[0]
# print(type(image_file))
# print(results['pred_class'])
# for i in range(len(image_file_list)):
# print(results['pred_class'])
# print(results['pred_score'])
