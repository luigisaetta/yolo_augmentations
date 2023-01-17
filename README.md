# YOLO Augmentations
This repo contains all the work done for the project YOLO Augmentations

## Introduction.
The goal of this project is to create a set of functions/tools to easily enable to augment your dataset for an **Object Detection** project
based on **YOLO V5/V8**.

These kind of tools are expecially useful if you have **few labeled images** to train your YOLO model.

## Technical specifications.
When we **annotate** a set of images to train an Object Detection (OD) model we need to define the region in the image containing an Object and the class of that object, for each of the Objects that you want to identify in the image.

In many OD models the region is a rectangle, called **Bounding Box Rectangle (BB)**, with sides parallel to x and y axes.

In YOLO v5 (and it is ok also for new v8) to register the annotation this is the format used:
1. For each image you have, in the labels directory, an associated txt file
2. The txt file contains a row for each BB rectangle
3. The format of each row is the following: class_num x y w h

for example:
```
2 0.6359375 0.37421875 0.03046875 0.03125
2 0.6359375 0.4203125 0.03828125 0.03671875
10 0.63671875 0.31953125 0.05703125 0.265625
```

One important thing to know is the way the rectangle is defined:
* x, y are the coordinates of the center, normalized (in other words, divided by width and height of the image in pixel)
* w and h are the width and the height of the BB, again normalized.

For this reason we have in the common_function.py module this code to get the BB to be used in cv2
```
# x lower left
l = int((x - w / 2.0) * width)
# x upper right
r = int((x + w / 2.0) * width)
# y lower left
t = int((y - h / 2.0) * height)
# y upper right
b = int((y + h / 2.0) * height)
```

    
## Features.
* functions to read/write YOLO v5 label files
* functions to switch from YOLO to CV2 BB format
* Augmentation based on Albumentations
* NB showing how to plot an image with BB

