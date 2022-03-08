import cv2
import numpy as np


class HOG():
    def __init__(self):
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9

    def get_frame_feature(self, image):
        image = image[0]
        image = np.array(image)
        image_h,image_w,_ = image.shape
        scale = 256 / float(max(image_w, image_h))
        self.ph = int(image_h * scale) // 4 * 4 + 4
        self.pw = int(image_w * scale) // 4 * 4 + 4
        self.winSize = (self.pw, self.ph)
        image = cv2.resize(image, (self.pw, self.ph))
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

        hist = self.hog.compute(image, self.winSize, padding = (0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return [hist.reshape(w, h, 36).transpose(2, 1, 0)] #36*36*64

    def normalize(n):
        max_val = np.max(n,axis=0).reshape(1,-1)
        min_val = np.min(n,axis=0).reshape(1,-1)
        return (n-min_val)/(max_val-min_val)

    def get_feature(self,image_list,target):
        final = []
        for image in image_list:
            image = np.array(image)
            image = cv2.resize(image, (self.pw, self.ph))
            winStride = self.winSize
            hist = self.hog.compute(image, winStride, padding = (0, 0))
            w, h = self.winSize
            sw, sh = self.blockStride
            w = w // sw - 1
            h = h // sh - 1
            #distance
            #feature = hist.reshape(w, h, 36).transpose(2, 1, 0)
            # dist = np.abs(feature - target)
            # rfc = np.sum(dist,axis=(1,2))/(dist.shape[1]*dist.shape[2])
            # final.append(rfc)

            #origin
            # feature = hist.reshape(w, h, 36)
            # feature = np.sum(feature,axis=2)/36
            # feature = feature.reshape(-1)
            # final.append(feature)

            # convolution
            feature = hist.reshape(w, h, 36).transpose(2, 1, 0)
            corelation = feature*target
            feature = np.sum(corelation,axis=(1,2))
            final.append(feature)
            


            

        return final