import numpy as np
from abc import abstractmethod
# import matplotlib.pyplot as plt
import copy
import os
import torch
from PIL import Image

# Author: hzj.2024.07.29
# Content: 1维2维检测器
# 三种门限选择：
    # CA，参考单元均值作为门限
    # SO，参考单元最小值为门限，isAll为True时选参考单元所有点的最小值，否则为左右两边均值中最小的为参考
    # GO，参考单元最大值为门限，isAll为True时选参考单元所有点的最大值，否则为左右两边均值中最大的为参考
    # 取所有点中最小（大）为常规逻辑，而取两边均值最小的策略参考： https://blog.csdn.net/weixin_45317919/article/details/125899133
    # 二维CFAR没有左右的问题，就直接选最小（大）为门限
# 门限系数通过get_factor函数确定，第一次计算时较慢，后续会直接读取第一次计算的结果(根据标准复高斯白噪声计算得到)

class NetMiddleware():
    DetectorType = ['CA']
    Detectors = None
    confusion_matrix = {}
    matrix_reset = {}
    Dims = 2
    saveImgPath = None
    Datanames  = None
    count = 1
    
    def __init__(self,detectorType = ['CA'], dims = 2, pf = 1e-6,ProLen = 2, RefLen = 5,names = None,saveImgPath = None):
        self.DetectorType = detectorType
        print('-----preparing detectors------')
        self.Detectors = [{1:CFAR_1D(dt,protectLen = ProLen,factor = 5,referenceLen = RefLen, pfa = pf),
                           2:CFAR_2D(dt,protectLen = ProLen,factor = 5,referenceLen = RefLen, pfa = pf)}[dims] for dt in detectorType]
        print('------finish preparing--------')
        self.confusion_matrix = {dt:torch.zeros(dims,dims).to(torch.int) for dt in detectorType}
        self.matrix_record = self.confusion_matrix
        self.Dims = dims
        self.saveImgPath = saveImgPath
        self.Datanames = names

        
        if not os.path.exists(saveImgPath):
            os.makedirs(saveImgPath)

    def set_names(self,names):
        self.Datanames = names

    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  num_class=2)
        return matrix.cpu().numpy()
    
    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        recall_by_class = {}
        recall = {}
        for key in self.confusion_matrix:
            confusion_matrix = self.confusion_matrix[key]
            one_recall_by_class = np.diag(confusion_matrix) / np.nansum(confusion_matrix, axis=1)
            one_recall_by_class = np.nan_to_num(one_recall_by_class)
            one_recall = np.mean(one_recall_by_class)
            recall_by_class[key] = one_recall_by_class
            recall[key] = one_recall
        return recall, recall_by_class

    def detect2D(self,data):
        b,_,_,_,_ = data.shape
        result = {}
        for k,d in enumerate(self.Detectors):
            detect_result = torch.zeros_like(data.squeeze()).to(torch.int)
            for i in range(b):
                a = data[i].to('cpu').squeeze()
                detect_result[i],_ = d.filter(a)
            result[self.DetectorType[k]] = detect_result
        return result

    def detect1D(self,data):
        b,_,_,w,_ = data.shape
        result = {}
        for k,d in enumerate(self.Detectors):
            detect_result = torch.zeros_like(data.squeeze(1,2)).to(torch.int)
            
            for i in range(b):
                a = data[i]
                for s in range(w):
                    c = a[:,:,s,:].to('cpu').squeeze()

                    detect_result[i,s,:],_ = d.filter(c)

            result[self.DetectorType[k]] = detect_result
        return result


    def update(self,data,mask):
        if 2 == self.Dims:
            self.update2D(data,mask)
        if 1 == self.Dims:
            self.update1D(data,mask)
    
    def update1D(self,data,mask):
        result = self.detect1D(data)
        for k in result:
            if self.saveImgPath is not None:
                self.saveImg(result[k],k)
            self.confusion_matrix[k] += self._generate_matrix(mask,result[k])
        
    def update2D(self,data,mask):
        result = self.detect2D(data)
        for k in result:
            self.confusion_matrix[k] += self._generate_matrix(mask,result[k])

    def get_pd_pf(self):
        _, recall_by_class = self.get_pixel_recall_class()
        pf = {}
        pd = {}
        for key in recall_by_class:
            a = recall_by_class[key]
            pf[key] = 1-a[0]
            pd[key] = a[1] #target is class 1
        return pd,pf

    def saveImg(self,data,detectType):
        if not os.path.exists(self.saveImgPath+'/'+detectType):
            os.makedirs(self.saveImgPath+'/'+detectType)

        for i in range(data.shape[0]):
            fig = Image.fromarray(np.array(data[i].cpu().clone(),dtype = np.uint8)*255)
            if self.Datanames is None:
                name = self.saveImgPath+'/'+detectType+'/'+str(self.count)+'.png'
            else:
                name = self.saveImgPath+'/'+detectType+'/'+self.Datanames[self.count][0]+'.png'
            fig.save(name)
            self.count += 1

    def remake(self):
        self.confusion_matrix = self.matrix_reset

class Detector():
    winset = None
    factor = None
    paraPath = './save/detectorPara'
    pfa = 1e-3

    def __init__(self,protectLen = 2,referenceLen = 3,winst = None,factor = None,pfa = 1e-3                                                                                                         ):
        if not os.path.exists(self.paraPath):
            os.makedirs(self.paraPath)
        self.pfa = pfa
        if winst is None:
            self.winset = self.consWin(protectLen,referenceLen)
        else:
            self.winset = winst
        
        if factor is None:
            self.get_factor()
        else:
            self.factor = factor
        

    @abstractmethod
    def consWin(self,protectLen,referenceLen):
        pass
    
    @abstractmethod
    def get_threshold(self,data):
        pass

    @abstractmethod
    def get_factor(self):
        pass

    def filter(self,data):
        data = torch.Tensor(abs(data))
        if self.factor is None:
            threshold = self.get_threshold(data)
        else:
            threshold = self.factor*self.get_threshold(data)
        result = copy.deepcopy(data)
        result[abs(result)<abs(threshold)] = 0
        result[abs(result)>=abs(threshold)] = 1
        return result,threshold

class CFAR_1D(Detector):
    CFARType = 'CA'
    ProtectLen = 0
    ReferenceLen = 3

    def __init__(self,CFARType = 'CA', protectLen=2, referenceLen=3, winst=None,factor = None,pfa = 1e-3):
        super().__init__(protectLen, referenceLen, winst,factor,pfa)
        self.CFARType = CFARType
        self.ProtectLen = protectLen
        self.ReferenceLen = referenceLen

    # def consWin(self,protectLen=2, referenceLen=3):
    #     refUnits = np.ones(referenceLen)
    #     proUnits = np.zeros(protectLen)
    #     win = np.concatenate((refUnits,proUnits,np.zeros(1),proUnits,refUnits))
    #     return win.squeeze()
        
    def consWin(self, protectLen = 2, referenceLen = 3):
        win = torch.ones(1+2*protectLen+2*referenceLen)
        win[referenceLen:referenceLen+2*protectLen+1] = 0
        return win
    
    def get_factor(self):
        
        path = self.paraPath+'/'+f'cfar1d_{self.CFARType}_{self.pfa}.npy'
        if os.path.exists(path):
            self.factor = np.load(path).item()
        else:
            
            num = int(1e2/self.pfa)
            noise = (np.random.randn(num)+1j*np.random.randn(num))*0.5**0.5
            _,threshold = self.filter(abs(noise))
            oc = abs(noise)/threshold.real
            oc.sort()
            self.factor = oc[-100]
            np.save(path,self.factor)
        

    def get_threshold(self, data):
        # 卷积实现加权平均
        return {'CA':self.get_CA_threshold,
                'SO':self.get_SO_threshold,
                'GO':self.get_GO_threshold
                }[self.CFARType](data)

    
    def get_CA_threshold(self,data):
        # data = data.squeeze()
        # win = self.winset[::-1] #反转
        win = self.winset.flip(0) #反转
        N = win.shape[0]+data.shape[0]-1
        result = torch.fft.ifft(torch.fft.fft(data,n=N)*torch.fft.fft(win,n=N))
        cout = torch.fft.ifft(torch.fft.fft(torch.ones_like(data),n=N)*torch.fft.fft(win,n=N))
        result /= cout
        return result[int(win.shape[0]/2):int(win.shape[0]/2)+data.shape[0]]
    
    def get_SO_threshold(self,data,isAll = False):
        threshold = torch.zeros_like(data).to(torch.float)
        L = len(data)
        for i in range(L):
            l = range(i-self.ProtectLen-self.ReferenceLen,i-self.ProtectLen)
            if (i+self.ProtectLen)<L and (i+self.ProtectLen+self.ReferenceLen+1)>=L:
                r = np.hstack((range(i+self.ProtectLen+1,L),range(0,(i+self.ProtectLen+self.ReferenceLen+1)%L))).astype(int)
            else:
                r = range((i+self.ProtectLen+1)%L,(i+self.ProtectLen+self.ReferenceLen+1)%L)
            if isAll:
                min_l = abs(data[l]).min()
                min_r = abs(data[r]).min()
            else:
                min_l = abs(data[l]).mean()
                min_r = abs(data[r]).mean()
            threshold[i] = min(min_l,min_r)
        return threshold
    
    def get_GO_threshold(self,data,isAll = False):
        threshold = torch.zeros_like(data).to(torch.float)
        L = len(data)
        for i in range(L):
            l = range(i-self.ProtectLen-self.ReferenceLen,i-self.ProtectLen)
            if (i+self.ProtectLen)<L and (i+self.ProtectLen+self.ReferenceLen)>=L:
                r = np.hstack((range(i+self.ProtectLen,L),range(0,(i+self.ProtectLen+self.ReferenceLen)%L))).astype(int)
            else:
                r = range((i+self.ProtectLen+1)%L,(i+self.ProtectLen+self.ReferenceLen+1)%L)
            if isAll:
                max_l = abs(data[l]).max()
                max_r = abs(data[r]).max()
            else:
                max_l = abs(data[l]).mean()
                max_r = abs(data[r]).mean()
            threshold[i] = max(max_l,max_r)
        return threshold

class CFAR_2D(Detector):
    ProtectLen = 0
    ReferenceLen = 3
    CFARType = 'CA'

    def __init__(self,CFARType = 'CA', protectLen=2, referenceLen=3, winst=None,factor = None,pfa= 1e-3):
        super().__init__( protectLen, referenceLen, winst,factor,pfa)
        self.CFARType = CFARType
        self.ProtectLen = protectLen
        self.ReferenceLen = referenceLen

    def consWin(self,protectLen=2, referenceLen=3):
        win = torch.ones((1+2*protectLen+2*referenceLen,1+2*protectLen+2*referenceLen))
        win[referenceLen:referenceLen+2*protectLen+1,referenceLen:referenceLen+2*protectLen+1] = 0
        return win
    
    def get_factor(self):
        path = self.paraPath+'/'+f'cfar2d_{self.CFARType}_{self.pfa}.npy'
        if os.path.exists(path):
            self.factor = np.load(path).item()
        else:
            num = int((1e2/self.pfa)**0.5)
            noise = (np.random.randn(num,num)+1j*np.random.randn(num,num))*0.5**0.5
            _,threshold = self.filter(abs(noise))
            oc = abs(noise)/threshold.real
            oc = oc.reshape(-1)
            oc,_ = oc.sort()
            self.factor = oc[-100]
            np.save(path,self.factor)

    def get_threshold(self, data):
        # 卷积实现加权平均
        return {'CA':self.get_CA_threshold,
                'SO':self.get_SO_threshold,
                'GO':self.get_GO_threshold
                }[self.CFARType](data)

    def get_CA_threshold(self,data):
        # win = self.winset[::-1].T[::-1].T #二维反转
        win = self.winset.flip(0,1)
        N,M = win.shape[0]+data.shape[0]-1,win.shape[1]+data.shape[1]-1
        result = torch.fft.ifft2(torch.fft.fft2(data,s=(N,M))*torch.fft.fft2(win,s=(N,M)))
        cout  = torch.fft.ifft2(torch.fft.fft2(torch.ones_like(data),s=(N,M))*torch.fft.fft2(win,s=(N,M)))
        result /= cout
        return result[int(win.shape[0]/2):int(win.shape[0]/2)+data.shape[0],int(win.shape[1]/2):int(win.shape[1]/2)+data.shape[1]]
    
    def get_SO_threshold(self,data):
        threshold = torch.zeros_like(data).to(torch.float)
        W,H = data.shape
        for wi in range(W):
            for hi in range(H):
                wl = 0 if wi-self.ProtectLen-self.ReferenceLen<0 else wi-self.ProtectLen-self.ReferenceLen
                wpl= 0 if wi-self.ProtectLen<0 else wi-self.ProtectLen
                wr = W if wi+self.ProtectLen+self.ReferenceLen+1>W else wi+self.ProtectLen+self.ReferenceLen+1
                wpr = W if wi+self.ProtectLen+1>W else wi+self.ProtectLen+1
                hl = 0 if hi-self.ProtectLen-self.ReferenceLen<0 else hi-self.ProtectLen-self.ReferenceLen
                hpl = 0 if hi-self.ProtectLen<0 else hi-self.ProtectLen
                hr = H if hi+self.ProtectLen+self.ReferenceLen+1>H else hi+self.ProtectLen+self.ReferenceLen+1
                hpr = H if hi+self.ProtectLen+1>H else hi+self.ProtectLen+1
                
                row_min = data[np.hstack((range(wl,wpl),range(wpr,wr))).astype(int),hl:hr].min() if len(np.hstack((range(wl,wpl),range(wpr,wr)))) else np.inf
                column_min = data[wl:wr,np.hstack((range(hl,hpl),range(hpr,hr))).astype(int)].min() if len(np.hstack((range(hl,hpl),range(hpr,hr)))) else np.inf

                threshold[wi,hi] = min(row_min,column_min)
        return threshold
        
    def get_GO_threshold(self,data):
        threshold = torch.zeros_like(data).to(torch.float)
        W,H = data.shape
        for wi in range(W):
            for hi in range(H):
                wl = 0 if wi-self.ProtectLen-self.ReferenceLen<0 else wi-self.ProtectLen-self.ReferenceLen
                wpl= 0 if wi-self.ProtectLen<0 else wi-self.ProtectLen
                wr = W if wi+self.ProtectLen+self.ReferenceLen+1>W else wi+self.ProtectLen+self.ReferenceLen+1
                wpr = W if wi+self.ProtectLen+1>W else wi+self.ProtectLen+1
                hl = 0 if hi-self.ProtectLen-self.ReferenceLen<0 else hi-self.ProtectLen-self.ReferenceLen
                hpl = 0 if hi-self.ProtectLen<0 else hi-self.ProtectLen
                hr = H if hi+self.ProtectLen+self.ReferenceLen+1>H else hi+self.ProtectLen+self.ReferenceLen+1
                hpr = H if hi+self.ProtectLen+1>H else hi+self.ProtectLen+1
                
                row_max = data[np.hstack((range(wl,wpl),range(wpr,wr))).astype(int),hl:hr].max() if len(np.hstack((range(wl,wpl),range(wpr,wr)))) else 0
                column_max = data[wl:wr,np.hstack((range(hl,hpl),range(hpr,hr))).astype(int)].max() if len(np.hstack((range(hl,hpl),range(hpr,hr)))) else 0

                threshold[wi,hi] = max(row_max,column_max)
        return threshold


def confusion_matrix(y_true, y_pred, num_class):
    N = num_class
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=y.dtype, device=y.device)))
    y = y.reshape(N, N)
    return y


def test_1d():
    det = CFAR_1D(CFARType = 'SO', referenceLen=2,protectLen=1)
    data = np.array([1,2,0,1,0,0,3])#输入data需要是abs
    result,threshold = det.filter(data)
    
    plt.figure
    plt.plot(range(7),data)
    plt.plot(range(7),threshold.real)
    plt.legend(['orign','threshold'])
    plt.show()

def test_2d():
    det = CFAR_2D('CA',referenceLen=1,protectLen=0,factor=4.3783)
    data = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]]) #输入data需要是abs
    result,threshold = det.filter(data)

    print(f'data is {data}')
    print(f'threshold is {threshold.real}')
    print(f'result is {result}')


if __name__ == '__main__':
    test_1d()
    # test_2d()
