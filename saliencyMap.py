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

# numpy for all calculations
import numpy as np
# for additional image calculations
from scipy import ndimage

#---------------------------------------------------------
# Original implementation from
# @incollection{Achanta,
#  author={Achanta, Radhakrishna and Estrada, Francisco and Wils, Patricia and Süsstrunk, Sabine},
#  editor={Gasteratos, Antonios and Vincze, Markus and Tsotsos, John K.},
#  booktitle={Computer Vision Systems},
#  title={Salient Region Detection and Segmentation},
#  year={2008},
#  pages={66--75},
#  volume={5008},
#  series={Lecture Notes in Computer Science},
#  publisher={Springer Berlin Heidelberg},
#  isbn={978-3-540-79546-9},
#  doi={10.1007/978-3-540-79547-6_7}
# }
# ported to directly Python/NumPy
#---------------------------------------------------------
def getSaliencyMapOriginal1(labImage):
  (height, width, channels) = labImage.shape
  md = min(width, height)
  l = labImage[:,:,0]
  a = labImage[:,:,1]
  b = labImage[:,:,2]
  off1 = np.round(md/2).astype(int)
  off2 = np.round(md/4).astype(int)
  off3 = np.round(md/8).astype(int)
  sm = np.zeros( shape=(height,width) )
  for j in range(0,height):
    y11 = max(0,j-off1)
    y12 = min(j+off1+1,height)
    y21 = max(0,j-off2)
    y22 = min(j+off2+1,height)
    y31 = max(0,j-off3)
    y32 = min(j+off3+1,height)
    for k in range(0,width):
      x11 = max(0,k-off1)
      x12 = min(k+off1+1,width)
      x21 = max(0,k-off2)
      x22 = min(k+off2+1,width)
      x31 = max(0,k-off3)
      x32 = min(k+off3+1,width)
      lm1 = np.mean(l[y11:y12,x11:x12])
      am1 = np.mean(a[y11:y12,x11:x12])
      bm1 = np.mean(b[y11:y12,x11:x12])
      lm2 = np.mean(l[y21:y22,x21:x22])
      am2 = np.mean(a[y21:y22,x21:x22])
      bm2 = np.mean(b[y21:y22,x21:x22])
      lm3 = np.mean(l[y31:y32,x31:x32])
      am3 = np.mean(a[y31:y32,x31:x32])
      bm3 = np.mean(b[y31:y32,x31:x32])
      cv1 = (l[(j,k)]-lm1)**2 + (a[(j,k)]-am1)**2 + (b[(j,k)]-bm1)**2
      cv2 = (l[(j,k)]-lm2)**2 + (a[(j,k)]-am2)**2 + (b[(j,k)]-bm2)**2
      cv3 = (l[(j,k)]-lm3)**2 + (a[(j,k)]-am3)**2 + (b[(j,k)]-bm3)**2
      sm[(j,k)]= cv1 + cv2 + cv3
  return sm

#---------------------------------------------------------
# As above, but row/column switched to numpy-friendly order (?)
# TODO: Check this is actually the numpy-friendly order!
#---------------------------------------------------------
def getSaliencyMapOriginal2(labImage):
  (height, width, channels) = labImage.shape
  md = min(width, height)
  l = labImage[:,:,0]
  a = labImage[:,:,1]
  b = labImage[:,:,2]
  off1 = np.round(md/2).astype(int)
  off2 = np.round(md/4).astype(int)
  off3 = np.round(md/8).astype(int)
  sm = np.zeros( shape=(height,width) )
  for k in range(0,width):
    x11 = max(0,k-off1)
    x12 = min(k+off1+1,width)
    x21 = max(0,k-off2)
    x22 = min(k+off2+1,width)
    x31 = max(0,k-off3)
    x32 = min(k+off3+1,width)
    for j in range(0,height):
      y11 = max(0,j-off1)
      y12 = min(j+off1+1,height)
      y21 = max(0,j-off2)
      y22 = min(j+off2+1,height)
      y31 = max(0,j-off3)
      y32 = min(j+off3+1,height)
      lm1 = np.mean(l[y11:y12,x11:x12])
      am1 = np.mean(a[y11:y12,x11:x12])
      bm1 = np.mean(b[y11:y12,x11:x12])
      lm2 = np.mean(l[y21:y22,x21:x22])
      am2 = np.mean(a[y21:y22,x21:x22])
      bm2 = np.mean(b[y21:y22,x21:x22])
      lm3 = np.mean(l[y31:y32,x31:x32])
      am3 = np.mean(a[y31:y32,x31:x32])
      bm3 = np.mean(b[y31:y32,x31:x32])
      cv1 = (l[(j,k)]-lm1)**2 + (a[(j,k)]-am1)**2 + (b[(j,k)]-bm1)**2
      cv2 = (l[(j,k)]-lm2)**2 + (a[(j,k)]-am2)**2 + (b[(j,k)]-bm2)**2
      cv3 = (l[(j,k)]-lm3)**2 + (a[(j,k)]-am3)**2 + (b[(j,k)]-bm3)**2
      sm[(j,k)]= cv1 + cv2 + cv3
  return sm

#---------------------------------------------------------
# Reimplementation of the algorithm above in a more 
# idiomatic manner for Python/NumPy
#---------------------------------------------------------
def getSaliencyMapNumpy(labImage, scales=3):
  (height, width, channels) = labImage.shape
  minimumDimension = min(width, height)

  # saliency map
  saliencyMap = np.zeros( shape=(height,width) )

  # calculate neighbourhood means for every scale and channel
  for s in range(0, scales):
    # TODO: make function for radius calculation parameter
    offset = np.round(minimumDimension/(2**(s+1))).astype(int)
    radius = offset*2+1
    # to correct the values near borders, see http://stackoverflow.com/questions/10683596/efficiently-calculating-boundary-adapted-neighbourhood-average
    filterMask = np.pad(np.ones((height,width)), offset, mode='constant', constant_values=0)
    filterFix = ndimage.uniform_filter(filterMask,radius,mode='constant',cval=0.0)
    filterFix = filterFix[offset:-offset, offset:-offset]
    for c in range(0, channels):
      # TODO: here R_1=1 is fixed (as in the original algorithm), make variable
      saliencyMap+=(labImage[:,:,c]-ndimage.uniform_filter(labImage[:,:,c],radius,mode='constant',cval=0.0)/filterFix)**2

  return saliencyMap

