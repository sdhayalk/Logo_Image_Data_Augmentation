# Logo Image Data Augmentation

The main aim of this project is to generate a new dataset from an existing one using an overlay image dataset (images in ```.png``` like logos) and a background image dataset (like ImageNet or MS COCO), and merging those two datasets to make a new dataset along with the normalized x, y coordinates, class number and label of the overlayed image on the background image.

I am using overlay images as logos in ```.png``` format and background images as ```.jpg``` files from MS COCO.

The overlay of logos is done randomly, with random selection of logos, random position, random rotation, random size, different types of random noise such as salt and pepper, etc. It also writes the following into separate file:
```
1) image file name
2) class label
3) class number
4) normalized min x coordinate
5) normalized min y coordinate
6) normalized max x coordinate
7) normalized max y coordinate
```

## Sample input and outputs:

Input Background Image:

![Input Background Image](Readme_Images/000000000092.jpg "<== Input Background Image")


Resultant Overlayed Image:

![Input Background Image](Readme_Images/000000000092overlayed.jpg "<== Resultant Overlayed Image")


Resultant Overlayed Label data:
```
000000000092.jpg,samsung,3,0.5078,0.3688,0.7312,0.4875
```


Resultant Overlayed Image with bounding box (just for reference):

![Input Background Image](Readme_Images/000000000092_.jpg "<== Resultant Overlayed Image with bounding box (just for reference)")


### TODO:
Add more augmentation techniques such as random shear, random lightening and color, etc.
