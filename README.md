# Semantic-Segmentation
Semantic Segmentation of an Image using Berkeley vision's caffe model. http://caffe.berkeleyvision.org


The segmented image gets classified into 21 predefined classes: ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
'diningtable', 'dog', 'horse', 'motorbike', 'person',
'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

Further a webpage is built to run the network locally for comparing the original and segmented image. 
Webpage.jpg shows an example for the output page

#To run the program
1. Make caffe locally http://caffe.berkeleyvision.org
2. Download the heavy caffemodel from the caffemodel-url in neural network folder
3. Install Flask Framework 
4. Run the network using flask_caller.py and visit 127.0.0.1:5000/upload on a browser.
