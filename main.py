import tensorflow as tf
import cv2
import os

from random import randint

class BackgroundImageManipulation:
	def __init__(self):
		pass

	def get_number_of_partitions(self):
		return randint(1, 3)

	def resize_background_image(self, image, d1, d2):
		resized_image = cv2.resize(image, (d1, d2))
		return resized_image


class LogoImageManipulation:
	# todo: return normalized coordinates so that it can be converted to VOC format
	def __init__(self):
		pass

	def random_resize(self, image, max_length, min_length=30):
		random_length = randint(min_length, max_length)
		ratio = float(float(image.shape[0]) / float(image.shape[1]))
		resized_image = cv2.resize(image, (random_length, int(ratio*random_length)))
		return resized_image

	def random_rotate(self, image, max_rotation_degree=30):
		image = tf.contrib.keras.preprocessing.image.random_rotation(image, max_rotation_degree, row_axis=0, col_axis=1, channel_axis=2)
		return image

	def random_position(self, image, max_dim_1):
		return randint(10, max_dim_1-10)


class OverlayLogoOnBackground(BackgroundImageManipulation, LogoImageManipulation):
	counter = 0

	def __init__(self):
		pass

	def blend_transparent(self, l_img, s_img):
		'''This function is directly copied and referred from Mateen Ulhaq's answer in https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
		Arguments:
			l_img {numpy array} -- the background image
			s_img {numpy array} -- the overlay image
		Returns:
			numpy array -- overlayed background image with overlay image
		'''
		x_offset=y_offset=1
		y1, y2 = y_offset, y_offset + s_img.shape[0]
		x1, x2 = x_offset, x_offset + s_img.shape[1]

		alpha_s = s_img[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s

		for c in range(0, 3):
		    l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
		                              alpha_l * l_img[y1:y2, x1:x2, c])

		return l_img

	def overlay(self, logo_image_path, background_image_path):
		self.background_image = cv2.imread(background_image_path)
		self.logo_image = cv2.imread(logo_image_path, -1)

		self.logo_image = self.random_resize(self.logo_image, max_length=300)	# random resize of logo
		self.logo_image = self.random_rotate(self.logo_image)
		cv2.imshow('l',  self.logo_image)
		cv2.waitKey(0)
		# self.logo_image = self.random_position(self.logo_image, self.background_image.shape[1] - self.logo_image.shape[1])

		self.overlayed_images = self.blend_transparent(self.background_image, self.logo_image)
		OverlayLogoOnBackground.counter += 1

		return self.background_image

	def overlay_and_write_to_disk(self, logo_image_path, background_image_path, write_path, file_name):
		overlayed_image = self.overlay(logo_image_path, background_image_path)
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)


def main():
	BACKGROUND_IMAGE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/background_images'
	LOGO_IMAGE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/logo_images'
	OVERLAYED_WRITE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/overlayed_images'
	DIM_1 = 256
	DIM_2 = 256
	
	if not os.path.exists(OVERLAYED_WRITE_PATH):
		os.makedirs(OVERLAYED_WRITE_PATH)

	background_images_list = os.listdir(BACKGROUND_IMAGE_PATH)
	logo_images_list = os.listdir(LOGO_IMAGE_PATH)
	NUMBER_OF_LOGOS = len(logo_images_list)

	overlay_generator = OverlayLogoOnBackground()
	
	for background_image_file_name in background_images_list:
		for logo_image_file_name in logo_images_list:
			logo_image = LOGO_IMAGE_PATH + os.sep + logo_image_file_name
			background_image = BACKGROUND_IMAGE_PATH + os.sep + background_image_file_name

			# background_image = overlay_generator.resize_background_image(background_image, (DIM_1, DIM_2))

			overlay_generator.overlay_and_write_to_disk(logo_image, \
														background_image, \
														OVERLAYED_WRITE_PATH, \
														background_image_file_name[0:-4] + str(OverlayLogoOnBackground.counter) + '.jpg')


if __name__ == '__main__':
	main()
