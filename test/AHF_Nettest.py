import torch.utils.data
from torch.utils.data import DataLoader
from libtiff import TIFF
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

#Device configuration
#循环验证
def to_tensor(image):
    image = np.array(image)
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image-min_i)/(max_i-min_i)
    return image
class MyDataset_double1(torch.utils.data.Dataset):
    def __init__(self, mimage1, mimage2,local,cut_size,
                 transform=None,target_transform=None):
        super(MyDataset_double1,self).__init__()
        self.cut_size = cut_size
        '''统一cutsize'''
        self.local = local
        self.image1 = mimage1
        self.image2 = mimage2

    def __getitem__(self, index):
        x_ms, y_ms, label = self.local[index]
        x_pan = int(4 * x_ms) #计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        self.image_ms = self.image1[:, x_pan:x_pan+self.cut_size,
                        y_pan:y_pan+self.cut_size]
        # print(self.image_ms)
        self.image_pan = self.image2[:, x_pan:x_pan+self.cut_size,
                          y_pan:y_pan+self.cut_size]
        # print(self.image_pan.shape)
        # print(self.image_pan)

        self.label = label

        return self.image_ms, self.image_pan, self.label

    def __len__(self):
        return len(self.local)
    
def mirror_add(image,cut_size):
    # cv2.BORDER_REPLICATE： 进行复制的补零操作， 只对边缘的点进行复制，然后该列上的点都是这些
    # cv2.BORDER_REFLECT: 进行翻转的补零操作，举例只对当前对应的边缘
    # gfedcba | abcdefgh | hgfedcb
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作， gfedcb | abcdefgh | gfedcb
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作
    # bcdegh | abcdefgh | abcdefg
    img = np.array(image)
    top_size, bottom_size, left_size, right_size = (int(cut_size/2), int(cut_size/2), int(cut_size/2), int(cut_size/2))
    # REPLICATE： 复制制最边缘上的一个点，所有的维度都使用当前的点
    REPLICATE = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    #print(REPLICATE.shape)
    return REPLICATE


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 指定GPU
repnumber=1
numclass = 11  # 最大类别数
print(numclass)
rootpath = '/home/newamax/Desktop/AHF-Net' # the path of the project

for rep1 in range(repnumber):#start from 0
    CUT_SIZE = 80  # '''修改1：图片切割尺寸统一为一个参数'''
    MS_CUT_SIZE = int(CUT_SIZE/4)
    test_rate= 0.2
    BATCH_SIZE = 64
    data_path_lab = rootpath+ '/data/label1.npy' 
    data_path_ms = rootpath+'/data/MS_F_cv.tif'
    data_path_pan =rootpath+'/data/pan_F_cv.tif'
    #读取数据和标签
    label = np.load(data_path_lab)
    label_h = label.shape[0]
    label_w = label.shape[1]
    ms = TIFF.open(data_path_ms, mode='r')
    pan = TIFF.open(data_path_pan, mode='r')
    image_ms = ms.read_image()
    image_pan = pan.read_image()
    m_ms = mirror_add(image_ms, CUT_SIZE)  # 多返回值函数在只返回一个返回元素时是元组
    m_pan = mirror_add(image_pan, CUT_SIZE)
    for i in range(1, numclass+1):
        index_I = np.where(label==i) #第i类坐标
        index_I = np.array(index_I).T
        len_I = len(index_I)  # 索引总长度
        len_test = int(len_I * test_rate)  # 第i类训练样本数
        index_test = np.arange(len_I) #建立第i类所有索引
        np.random.shuffle(index_test)  # 打乱索引顺序
        label_train_i = i * np.ones((len_test, 1), dtype='int64')  # 第i类训练样本label
        if i == 1:
            test_data_label = label_train_i
            test_data_loca = index_I[index_test[:len_test]]

        else:
            test_data_label=np.append(test_data_label, label_train_i, axis=0)
            test_data_loca=np.append(test_data_loca, index_I[index_test[:len_test]], axis=0)

    
    test_data_label = test_data_label - 1  # label要从0开始
    test_data_loca = np.hstack((test_data_loca, test_data_label))
    true_label = test_data_loca[:, 2]
    pred_label = np.zeros(len(true_label), dtype=int)

    print(test_data_loca.shape)
    np.random.shuffle(test_data_loca)
    #数据归一化
    m_ms = to_tensor(m_ms)
    m_pan = to_tensor(m_pan)
    m_pan = np.expand_dims(m_pan, axis=0) # 二维数据进网络前要加一维
    m_ms = np.array(m_ms).transpose((2,0,1)) #调整通道
    ##转换类型
    m_ms = torch.from_numpy(m_ms).type(torch.FloatTensor)
    m_pan = torch.from_numpy(m_pan).type(torch.FloatTensor)
    test_data_loca = torch.from_numpy(test_data_loca).type(torch.LongTensor)
    
    Net = torch.load(rootpath+'/model/model.pkl')
    Net.to(device)
    Net.eval()

    test_data_real = MyDataset_double1(m_ms, m_pan, test_data_loca, cut_size=CUT_SIZE)
    test_loader_real = DataLoader(test_data_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    loc = 0
    for j, (ms, pan, label) in enumerate(test_loader_real):
        batch_t1 = ms.to(device)
        batch_t2 = pan.to(device)
        test_label = label.to(device)
        with torch.no_grad():
            test_output_real = Net(batch_t1, batch_t2)
        test_out_data_real = test_output_real.data.cpu().numpy()
        test_result_real = np.argmax(test_out_data_real, axis=1)
        b, h, w, c = batch_t1.shape
        for k, cls in enumerate(test_result_real):
            x1, y1, lab1 = test_data_loca[k + b * j]
            pred_label[loc] = cls
            loc = loc + 1



    cm = confusion_matrix(true_label, pred_label,labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(['0', '1', '2', '3', '4', '5', '6','7', '8', '9', '10'])
    print(cm)
    dsa=0#对角线元素
    pe = 0#计算kappa系数
    aa = 0
    aai = 0
    ses = np.sum(cm,axis=1)
    sep = np.sum(cm,axis=0)
    print("\n标记各类样本数：",ses,
          '\n预测各类样本数：',sep)
    print(np.sum(ses), np.sum(sep))
    for i, cla in enumerate(cm):
        pred = cla[i] #第i类预测正确总数
        dsa+=cla[i] #所有类预测类别总数cm对角线之和
        pe+=sep[i]*ses[i]
        acr = pred / sep[i]
        aa+=acr
        rep = pred / ses[i]
        f1 = 2*acr*rep / (acr + rep)
        print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f" % (i, acr, rep, f1))
    total_e=np.sum(cm)
    p=dsa/total_e
    aa = aa/numclass
    pe=pe/(total_e*total_e)
    kappa = (p-pe)/(1-pe)
    print("总体准确率: %.7f || 平均准确率： %.7f || kappa: %.7f" % (p,aa, kappa))




