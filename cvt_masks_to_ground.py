import cv2
import os
import numpy as np

# loading image files
basicImgsPath = "./data/masks_full/"
selectImgsPath = os.path.join(basicImgsPath)
imgFileList = os.listdir(selectImgsPath)
imgFileListJpgWext= [file for file in imgFileList if file.endswith(".jpg")]
imgFileListJpgWextSorted = sorted(imgFileListJpgWext)

# target image path
targetImgsPath = "./data/masks/"
selectDstsPath = os.path.join(targetImgsPath)

nCount = 0
while True:
  imgSrc = cv2.imread(selectImgsPath + imgFileListJpgWextSorted[nCount], cv2.IMREAD_COLOR)
  height, width, channel = imgSrc.shape
  imgDst = np.zeros((height, width), np.uint8)

  for i in range(height):
    for j in range(width):
      b = imgSrc.item(i, j, 0)
      g = imgSrc.item(i, j, 1)
      r = imgSrc.item(i, j, 2)
      
      if b <= 10 and r <= 10:
        if g >= 70:
          imgDst.itemset(i, j, 2)
          # imgDst.itemset((i, j, 0), 1)
        else:
          imgDst.itemset(i, j, 1)
      else:
        imgDst.itemset(i, j, 1)
            

  fileInfo = os.path.splitext(imgFileListJpgWextSorted[nCount])
  fileDstPath = selectDstsPath + fileInfo[0] + ".jpg"
  cv2.imwrite(fileDstPath, imgDst)
  print("curr:", fileDstPath, nCount)
  # cv2.imshow("imgDst", imgDst)
  # cv2.waitKey(0)
  nCount += 1
