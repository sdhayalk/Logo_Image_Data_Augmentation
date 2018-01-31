import cv2
import os
import random

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
		x_offset = randint(1, l_img.shape[1] - s_img.shape[1])
		y_offset = randint(1, l_img.shape[0] - s_img.shape[0])
		y1, y2 = y_offset, y_offset + s_img.shape[0]
		x1, x2 = x_offset, x_offset + s_img.shape[1]

		alpha_s = s_img[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s

		for c in range(0, 3):
		    l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
		                              alpha_l * l_img[y1:y2, x1:x2, c])

		return l_img

	def overlay(self, logo_image, background_image):
		self.background_image = background_image
		self.logo_image = logo_image

		self.logo_image = self.random_resize(self.logo_image, max_length=300)	# random resize of logo
		self.logo_image = self.random_rotate(self.logo_image)
		# self.logo_image = self.random_position(self.logo_image, self.background_image.shape[1] - self.logo_image.shape[1])

		self.overlayed_images = self.blend_transparent(self.background_image, self.logo_image)

		return self.background_image

	def overlay_and_write_to_disk(self, logo_image, background_image, write_path, file_name):
		OverlayLogoOnBackground.counter += 1
		overlayed_image = self.overlay(logo_image, background_image)
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)

	def write_to_disk(self, overlayed_image, write_path, file_name):
		OverlayLogoOnBackground.counter += 1
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
		background_image = cv2.imread(BACKGROUND_IMAGE_PATH+os.sep+background_image_file_name)
		temp_number_of_logos = randint(1, 3)

		for _ in range(temp_number_of_logos):
			logo_image_file_name = random.choice(logo_images_list)
			logo_image = cv2.imread(LOGO_IMAGE_PATH+os.sep+logo_image_file_name, -1)

			background_image = overlay_generator.overlay(logo_image, background_image)

		overlay_generator.write_to_disk(background_image, OVERLAYED_WRITE_PATH, background_image_file_name[0:-4] + str(OverlayLogoOnBackground.counter) + '.jpg')


if __name__ == '__main__':
	main()
