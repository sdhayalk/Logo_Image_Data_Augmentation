import cv2
import os

class OverlayLogoOnBackground:
	def __init__(self):
		pass

	def overlay(self, logo_image_path, background_image_path):
		self.background_image = cv2.imread(background_image_path)
		self.logo_image = cv2.imread(logo_image_path)

		self.overlayed_image = cv2.addWeighted(self.background_image, 1.0, self.logo_image, 0.0, 0)
		return self.overlayed_image

	def overlay_and_write_to_disk(self, logo_image_path, background_image_path, write_path, file_name):
		overlayed_image = self.overlay(logo_image_path, background_image_path)
		cv2.imwrite(write_path + os.sep + file_name, overlayed_image)


def main():
	pass

if __name__ == '__main__':
	main()