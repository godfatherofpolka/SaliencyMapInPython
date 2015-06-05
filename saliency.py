#!/usr/bin/env python

'''
Copyright 2015 Samuel Bucheli

This file is part of SaliencyMapInPython.

SaliencyMapInPython is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

SaliencyMapInPython is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SaliencyMapInPython.  If not, see <http://www.gnu.org/licenses/>.
'''

# for handling command line arguments
import argparse
# read, display, and save the images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# numpy!
import numpy as np
# for color space conversion
from skimage.color import lab2rgb,rgb2lab
# for some simple profiling
import time


# the actual functions
from saliencyMap import getSaliencyMapOriginal1,getSaliencyMapOriginal2,getSaliencyMapNumpy


#---------------------------------------------------------
# Main entry point, loads image, etc.
# Sample usage:
# $ python saliency.py -v test/test.png saliencyMap.png
#---------------------------------------------------------
def main():
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Calculate saliency map')
  parser.add_argument('input', help='input image')
  parser.add_argument('output', help='output file')
  parser.add_argument('-v', '--view', help='display image', action='store_true')
  args = parser.parse_args()

  # read image
  rgbImage = mpimg.imread(args.input)

  # convert to lab
  labImage = rgb2lab(rgbImage)
  # TODO: Matlab scales/shifts values, so we do the same in order to compare results
  labImage[:,:,0]=labImage[:,:,0]*2.55
  labImage[:,:,1]=labImage[:,:,1]+128
  labImage[:,:,2]=labImage[:,:,2]+128

  # TODO: If you want to compare things to the original implementation
  '''start = time.clock()
  sm1 = getSaliencyMapOriginal1(labImage)
  end = time.clock()
  print "getSaliencyMapOriginal1() took ", (end-start), " seconds"

  start = time.clock()
  sm2 = getSaliencyMapOriginal2(labImage)
  end = time.clock()
  print "getSaliencyMapOriginal2() took ", (end-start), " seconds"'''
  
  start = time.clock()
  #calculate saliency map
  sm3 = getSaliencyMapNumpy(labImage)
  end = time.clock()
  print "getSaliencyMapNumpy() took", (end-start), " seconds"
  
  output = sm3

  # TODO: you can use this if you want to check the output does not differ too much (modulo small floating point rounding errors) from the original implementation
  '''diff = np.sum((sm3-sm2)**2)
  print "Square-sum of difference: ", diff'''
  
  # TODO: implement clustering/segmentation
 
  # display output, if requested
  if args.view:
    plt.imshow(output, cmap = cm.Greys)
    plt.show()

  # save output
  mpimg.imsave(args.output,output)
  
if __name__ == "__main__":
  main()
