import cv2
import os

class OverlayLogoOnBackground:
	counter = 0
	def __init__(self):
		pass

	def overlay(self, logo_image_path, background_image_path):
		self.background_image = cv2.imread(background_image_path)
		self.logo_image = cv2.imread(logo_image_path)
		print(logo_image_path)
		
		# referenced from: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
		# https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
		self.overlayed_image = cv2.addWeighted(self.background_image, 1.0, self.logo_image, 0.0, 0)	
		OverlayLogoOnBackground.counter += 1

		return self.background_image

	def overlay_and_write_to_disk(self, logo_image_path, background_image_path, write_path, file_name):
		overlayed_image = self.overlay(logo_image_path, background_image_path)
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)


def main():
	BACKGROUND_IMAGE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/background_images'
	LOGO_IMAGE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/logo_images'
	OVERLAYED_WRITE_PATH = 'G:/DL/Logo_Image_Data_Augmentation/Images/overlayed_images'
	
	if not os.path.exists(OVERLAYED_WRITE_PATH):
		os.makedirs(OVERLAYED_WRITE_PATH)

	background_images_list = os.listdir(BACKGROUND_IMAGE_PATH)
	logo_images_list = os.listdir(LOGO_IMAGE_PATH)

	overlay_generator = OverlayLogoOnBackground()
	
	for background_image_file_name in background_images_list:
		for logo_image_file_name in logo_images_list:
			overlay_generator.overlay_and_write_to_disk(LOGO_IMAGE_PATH + os.sep + logo_image_file_name, \
														BACKGROUND_IMAGE_PATH + os.sep + background_image_file_name, \
														OVERLAYED_WRITE_PATH, \
														background_image_file_name[0:-4] + str(OverlayLogoOnBackground.counter) + '.jpg')

if __name__ == '__main__':
	main()