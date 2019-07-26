# import the necessary packages
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
 
# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's static saliency spectral residual detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

cv2.imwrite(args['image'][:-4]+'_cv2_saliency.jpg', saliencyMap.astype(int))	
#cv2.imshow("Image", image)
#cv2.imshow("Output", saliencyMap)
#cv2.waitKey(0)
