# just keep the celebrity who has picture under age 18
import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
from align_mtcnn.mtcnn_detector import MtcnnDetector

def get_cel_name(splited_arr):
	cel_name = ""
	for i in splited[1:-1]:
		cel_name += i
	return cel_name	


def crop_align_face(face, file_path_save):
	ctx = mx.cpu()
	mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
	mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
	count_no_find_face = 0
	count_crop_images = 0
	face_img = cv2.imread(face)
	ret = mtcnn.detect_face(face_img)
	if ret is None:
		print('%s do not find face'%file_path)
		count_no_find_face += 1
		return
	bbox, points = ret
	# print(bbox, points)
	for i in range(bbox.shape[0]):
		bbox_ = bbox[i, 0:4]
		points_ = points[i, :].reshape((2, 5)).T
		face = mtcnn.preprocess(face_img, bbox_, points_, image_size="224")
		cv2.imwrite(file_path_save, face)
		# cv2.imshow('face', face)
		# cv2.waitKey(0)
	count_crop_images += 1
	print('%d images crop successful!' % count_crop_images)

def mtcnn(pic_name):
	face_img = cv2.imread(file_path)


output_root = "output_cacd_has_under18"
root = "CACD2000"
if not os.path.exists(output_root):
	os.mkdir(output_root)

for root, dirs, files in os.walk(root):
	celebrity = set()
	picname_arr = []
	# name is pic name not celebrity's name
	# build arr age ascend
	for name in files:
		picname_arr.append(name)
	picname_arr.sort()
	print(picname_arr)

	for pic_name in picname_arr:
		splited = pic_name.split("_")
		cel_name = get_cel_name(splited)
		file_path = os.path.join(root, pic_name)
		file_path_save = os.path.join(output_root, pic_name)
		if cel_name in celebrity:
			print(file_path_save)
			crop_align_face(file_path, file_path_save)
			continue
		age = int(splited[0])
		if age <= 20 and age>18:
			celebrity.add(cel_name)
			print(file_path_save)
			crop_align_face(file_path, file_path_save)



		

	

