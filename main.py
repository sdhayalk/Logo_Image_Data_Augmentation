import cv2
import os
import random
import numpy as np

from random import randint

class BackgroundImageManipulation:
	def __init__(self, DIM_1, DIM_2):
		'''Constructor for class BackgroundImageManipulation
		Arguments:
			DIM_1 {number} -- dimension 1 of background image
			DIM_2 {number} -- dimension 2 of background image
		'''
		self.DIM_1 = DIM_1
		self.DIM_2 = DIM_2


class LogoImageManipulation:
	def __init__(self):
		pass

	def random_resize(self, image, max_length, min_length=30):
		'''This function randomly resizes the images, in range [min_length, max_length]
		Arguments:
			image {numpy array} -- image to be resized
			max_length {number} -- max length of the resize
		Keyword Arguments:
			min_length {number} -- min length of the resize (default: {30})
		Returns:
			numpy array -- randomly resized image
		'''
		random_length = randint(min_length, max_length)
		ratio = float(float(image.shape[0]) / float(image.shape[1]))
		resized_image = cv2.resize(image, (random_length, int(ratio*random_length)))
		return resized_image

	def random_noise(self, image):
		'''This function is direcly copied from Shubham Pachori's answer at: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
		Arguments:
			image {[type]} -- [description]
		Returns:
			[type] -- [description]
		'''
		noise_typ = randint(0, 4)	# 0 for adding no noise
		if noise_typ == 1: # "gauss"
			row,col,ch= image.shape
			mean = 0
			var = 0.1
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			noisy = image + gauss
			return noisy

		elif noise_typ == 2: # "s&p"
			row,col,ch = image.shape
			s_vs_p = 0.5
			amount = 0.004
			out = np.copy(image)
			# Salt mode
			num_salt = np.ceil(amount * image.size * s_vs_p)
			coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
			out[coords] = 1

			# Pepper mode
			num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
			coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
			out[coords] = 0

			return out

		elif noise_typ == 3: # "poisson"
			vals = len(np.unique(image))
			vals = 2 ** np.ceil(np.log2(vals))
			noisy = np.random.poisson(image * vals) / float(vals)
			return noisy

		elif noise_typ == 4: # "speckle"
			row,col,ch = image.shape
			gauss = np.random.randn(row,col,ch)
			gauss = gauss.reshape(row,col,ch)        
			noisy = image + image * gauss
			return noisy

	def random_rotate(self, image, max_rotation_degree=30):
		'''This function rotates the images randomly with degree range [-max_rotation_degree, +max_rotation_degree]
		Arguments:
			iamge {numpy array} -- the image to be randomly rotated
			max_rotation_degree {numpy array} -- random rotatio limit; default=30
		Returns:
			numpy array -- randomly rotated image
		'''
		angle = randint(-max_rotation_degree, max_rotation_degree)

		'''
		this logic has been directly referred from Remi Cuingnet's answer here at: https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/37347070#37347070
		'''
		height, width = image.shape[:2]
		image_center = (width/2, height/2)

		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

		abs_cos = abs(rotation_mat[0,0])
		abs_sin = abs(rotation_mat[0,1])

		bound_w = int(height * abs_sin + width * abs_cos)
		bound_h = int(height * abs_cos + width * abs_sin)

		rotation_mat[0, 2] += bound_w/2 - image_center[0]
		rotation_mat[1, 2] += bound_h/2 - image_center[1]

		rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
		return rotated_mat

	def random_position(self, image, max_dim_1):
		return randint(10, max_dim_1-10)


class OverlayLogoOnBackground(BackgroundImageManipulation, LogoImageManipulation):
	counter = 0

	def __init__(self, DIM_1, DIM_2):
		'''Constructor
		Arguments:
			DIM_1 {number} -- dimension 1 of background image
			DIM_2 {number} -- dimension 2 of background image
		'''
		BackgroundImageManipulation.__init__(self, DIM_1, DIM_2)

	def blend_transparent(self, l_img, s_img):
		'''This function is directly copied and referred from Mateen Ulhaq's answer in https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
		Arguments:
			l_img {numpy array} -- the background image
			s_img {numpy array} -- the overlay image
		Returns:
			numpy array -- overlayed background image with overlay image
		'''
		x_offset = randint(1, l_img.shape[1] - s_img.shape[1])
		y_offset = randint(1, l_img.shape[0] - s_img.shape[0])
		y1, y2 = y_offset, y_offset + s_img.shape[0]
		x1, x2 = x_offset, x_offset + s_img.shape[1]

		alpha_s = s_img[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s

		for c in range(0, 3):
		    l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
		                              alpha_l * l_img[y1:y2, x1:x2, c])

		return l_img, float(x1)/float(self.DIM_1), float(y1)/float(self.DIM_2), float(x2)/float(self.DIM_1), float(y2)/float(self.DIM_2)

	def overlay(self, logo_image, background_image):
		'''this functions (randomly) overlays the logo image with 0 percent transparency on the background image
		Arguments:
			logo_image {numpy array} -- the logo image
			background_image {numpy array} -- the background image
		Returns:
			numpy array -- returns background image with logo overalayes
			x_min -- normalized min x coordinate
			y_min -- normalized min y coordinate
			x_max -- normalized max x coordinate
			y_max -- normalized max y coordinate
		'''
		self.background_image = background_image
		self.logo_image = logo_image

		self.logo_image = self.random_resize(self.logo_image, max_length=300)	# random resize of logo
		self.logo_image = self.random_rotate(self.logo_image)
		self.logo_image = self.random_noise(self.logo_image)

		self.overlayed_images, x_min, y_min, x_max, y_max = self.blend_transparent(self.background_image, self.logo_image)

		return self.background_image, x_min, y_min, x_max, y_max

	def overlay_and_write_to_disk(self, logo_image, background_image, write_path, file_name):
		'''this functions (randomly) overlays the logo image with 0 percent transparency on the background image and writes it to disk
		Arguments:
			logo_image {numpy array} -- the logo image
			background_image {numpy array} -- the background image
			write_path {str} -- the path where to write the overlayed image
			file_name {str} -- name of the file to be given to the overlayed image
		'''
		OverlayLogoOnBackground.counter += 1
		overlayed_image, x_min, y_min, x_max, y_max = self.overlay(logo_image, background_image)
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)

	def write_image_to_disk(self, overlayed_image, write_path, file_name):
		'''this function writes overlayed_image to disk
		Arguments:
			overlayed_image {numpy array} -- the overlayed image
			write_path {str} -- the path where to write the overlayed image
			file_name {str} -- name of the file to be given to the overlayed image
		'''
		OverlayLogoOnBackground.counter += 1
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)

	def write_label_data_to_disk(self, overlayed_image_file_name, write_path, class_value, class_index_map, x_min, y_min, x_max, y_max):
		'''this function writes the label data to disk
		Arguments:
			overlayed_image_file_name {str} -- name of the overlayed image file 
			write_path {str} -- the path where to write the overlayed image
			class_value {str} -- class label of the logo
			class_index_map {dict} -- dictionary which maps class in string (key) to a numeric value (value)
			x_min {float} -- normalized min x coordinate
			y_min {float} -- normalized min y coordinate
			x_max {float} -- normalized max x coordinate
			y_max {float} -- normalized max y coordinate
		'''
		class_value = class_value[0:-1] 	# removing the last letter which represents the count of the same label
		class_index = class_index_map[class_value]

		with open(write_path + os.sep + overlayed_image_file_name + '.txt','w') as file:	
			file.write("{},{},{},{},{},{},{}".format(overlayed_image_file_name + '.jpg', \
											   class_value,
											   str(class_index),
											   "{0:.4f}".format(x_min), 
											   "{0:.4f}".format(y_min),
											   "{0:.4f}".format(x_max),
											   "{0:.4f}".format(y_max)))


def main():
	BACKGROUND_IMAGE_PATH = 'G:/DL/data_logo/coco_data/train2017'
	LOGO_IMAGE_PATH = 'G:/DL/data_logo/coco_data/logo_images'
	OVERLAYED_WRITE_PATH = 'G:/DL/data_logo/coco_data/overlayed_images'
	DIM_1 = 640
	DIM_2 = 480
	
	if not os.path.exists(OVERLAYED_WRITE_PATH):
		os.makedirs(OVERLAYED_WRITE_PATH)

	background_images_list = os.listdir(BACKGROUND_IMAGE_PATH)
	logo_images_list = os.listdir(LOGO_IMAGE_PATH)
	NUMBER_OF_LOGOS = len(logo_images_list)

	class_index_map = {}
	class_index_counter = 0
	for file_name in logo_images_list:
		if file_name[0:-5] not in class_index_map:
			print(file_name[0:-5])
			class_index_map[file_name[0:-5]] = class_index_counter
			class_index_counter += 1
	print('class_index_map:', class_index_map)

	overlay_generator = OverlayLogoOnBackground(DIM_1, DIM_2)
	
	loop_counter = 0
	for background_image_file_name in background_images_list[0:100]:
		try:
			background_image = cv2.imread(BACKGROUND_IMAGE_PATH+os.sep+background_image_file_name)
			temp_number_of_logos = randint(1, 3)

			background_image = cv2.resize(background_image, (DIM_1, DIM_2))

			# for _ in range(temp_number_of_logos):
			logo_image_file_name = random.choice(logo_images_list)
			logo_image = cv2.imread(LOGO_IMAGE_PATH+os.sep+logo_image_file_name, -1)

			background_image, x_min, y_min, x_max, y_max = overlay_generator.overlay(logo_image, background_image)

			# overlay_generator.write_image_to_disk(background_image, OVERLAYED_WRITE_PATH, background_image_file_name[0:-4] + str(OverlayLogoOnBackground.counter) + '.jpg')
			overlay_generator.write_image_to_disk(background_image, OVERLAYED_WRITE_PATH+os.sep+'Images', background_image_file_name)
			overlay_generator.write_label_data_to_disk(background_image_file_name[0:-4], OVERLAYED_WRITE_PATH+os.sep+'Labels', logo_image_file_name[0:-4], class_index_map, x_min, y_min, x_max, y_max)

			loop_counter += 1

		except:
			print('Error found at loop counter')

if __name__ == '__main__':
	main()
