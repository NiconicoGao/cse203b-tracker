import numpy as np
from PIL import Image
from resnet import NNfeatures
import cv2
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA


class MyTracker():
    def __init__(self, model_name='resnet18'):
        self.a=1
        if model_name == 'resnet18':
            self.Resnet = NNfeatures('resnet18')
        if model_name == 'resnet34':
            self.Resnet = NNfeatures('resnet34')
        if model_name == 'resnet50':
            self.Resnet = NNfeatures('resnet18')
        if model_name == 'resnet101':
            self.Resnet = NNfeatures('resnet101')
        self.krrs = []
        

    def init(self, image, roi):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h,image_w,_ = image.shape
        x1, y1, w, h = roi
        cx, cy, sub_image = self.get_sub_image(image, x1, y1, w, h,scale=1.5)
        self.frame = self.Resnet.get_frame_feature([sub_image])[0]

        target = self.gaussian_peak(w,h)
        summation = np.max(target)
        target = target/summation

        self.train_x,self.train_y,debug = self.get_train_data(image,target,w,h,cx,cy)
        for i in range(len(self.train_x)):
            gx = np.random.randint(0,image_w*0.7)+image_w*0.15
            gy = np.random.randint(0,image_h*0.7)+image_h*0.15
            gx,gy = int(gx),int(gy)
            if not self.in_roi((gx,gy),roi) and not self.in_roi((gx+w,gy+h),roi):
                _,_,neg = self.get_sub_image(image,gx,gy,w,h)
                self.train_x.append(neg)
                self.train_y.append(0)
        # plt.imshow(target, cmap='hot', interpolation='nearest')
        # plt.show()
        
        # plt.imshow(train_x[0])
        # plt.show()
        RFC = self.Resnet.get_feature(self.train_x,self.frame) #(36*256*14*14)
        # RFC = self.Resnet.get_feature2(train_x) #(36*256*14*14)
        # pca = PCA(n_components=32)
        # RFC = pca.fit_transform(RFC)
        krr = KernelRidge(alpha=1.0)
        self.krrs.append(krr)
        self.krrs[-1].fit(RFC, np.array(self.train_y))
        
        if len(self.krrs) > 4:
            self.krrs = self.krrs[1:]
        self.roi = roi

    def in_roi(self,point,roi):
        x,y = point
        x1,y1,w,h = roi
        if x1<x<x1+w and y1<y<y1+h:
            return True
        return False


    def get_sub_image(self, image, x1, y1, w, h,scale=1):
        image_h,image_w,_ = image.shape
        cx = x1 + w // 2
        cy = y1 + h // 2
        w = int(min(scale*w,image_w))
        h = int(min(scale*h,image_h))
        x = int(cx - w // 2) if int(cx - w // 2)>=0 else 0
        x = image_w-w if x+w>image_w else x
        y = int(cy - h // 2) if int(cy - h // 2)>=0 else 0
        y = image_h-h if y+h>image_h else y
        sub_image = image[y:y+h, x:x+w, :]
        sub_image= Image.fromarray(sub_image)
        return cx,cy,sub_image
    
    def gaussion_random(self,mu,sigma):
        s = np.random.normal(mu, sigma, 1)
        return int(s.item(0))

    def get_random(self,len):
        s = np.random.random(1)-0.5
        return s.item(0)*len

    def get_train_data(self,image,target,w,h,cx,cy,num=36):
        train_x = []
        train_y = []
        debug = []
        image_h,image_w,_= image.shape
        for i in range(num):
            dx = self.get_random(min(w,cx,image_w-cx))
            dy = self.get_random(min(h,cy,image_h-cy))
            tx = cx + dx
            ty = cy + dy
            x = int(tx - w // 2) if int(tx - w // 2)>=0 else 0
            y = int(ty - h // 2) if int(ty - h // 2)>=0 else 0
            

            sub_image = image[y:y+h, x:x+w, :]
            train_image = cv2.resize(sub_image, (w, h))
            

            target_h,target_w = target.shape
            target_x,target_y = int(target_w/2+dx),int(target_h//2+dy)
            if 0 <= target_x < target_w and 0 <= target_y < target_h:
                train_y.append(target[target_y][target_x])
                train_x.append(train_image)
                debug.append((dx,dy,cx,cy))

        train_x = [Image.fromarray(image) for image in train_x]

        return train_x,train_y,debug


    def update(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, w, h = self.roi
        cx, cy, sub_image = self.get_sub_image(image, x1, y1, w, h,scale=1.5)
        scale = [(0.85,0.85),(1,1),(1.05,1.05)]
        test_x = []
        roi_set = []
        for s in scale:
            for i in range(15):
                th,tw = int(h*s[0]),int(w*s[1])
                n_x1 = self.gaussion_random(x1,tw/6)
                n_y1 = self.gaussion_random(y1,th/6)
                cx, cy, sub_image = self.get_sub_image(image, n_x1, n_y1, tw, th,scale=1)
                test_x.append(sub_image)
                roi_set.append((n_x1,n_y1,tw,th))

        RFC = self.Resnet.get_feature(test_x,self.frame) #(36*256*14*14)
        # RFC = np.array(RFC)
        # pca = PCA(n_components=32)
        # RFC = pca.fit_transform(RFC)
        result = self.krrs[0].predict(RFC)
        result = np.array(result)
        for k in range(1, len(self.krrs)):
            result = 1/9 * result + self.krrs[k].predict(RFC)
        index = np.argmax(result)
        self.roi = roi_set[index]
        

        return self.roi

    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh+h, -sxh:-sxh+w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x**2 + y**2)/(2. * sigma**2)))
        return g
