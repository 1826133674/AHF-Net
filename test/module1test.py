import time
import tifffile
import datetime as dt
import cv2
from AWIHS import dfusion
start = time.time()
rootpath = '/home/newamax/Desktop/AHF-Net/data'
data_path_pan = rootpath+'/pan.tif'
data_path_ms_bic = rootpath+'/ms4.tif'
#读取数据和标签\
image_ms = tifffile.imread(data_path_ms_bic)
image_pan = tifffile.imread(data_path_pan)
# 融合图像
start_time = dt.datetime.now().strftime('%F %T')
print("程序开始运行时间：" + start_time)
fx = 4
fy = 4
image_ms_Bic=cv2.resize(image_ms,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC)
MS_F, pan_F= dfusion(image_ms_Bic,image_pan)
data_path_pan_F = rootpath+'/pan_F_cv.tif'
tifffile.imsave(data_path_pan_F, pan_F)
data_path_ms_F =rootpath+'/MS_F_cv.tif'
tifffile.imsave(data_path_ms_F, MS_F)
end = time.time()
print("The running time is : %.2f s" % (end-start))
end_time = dt.datetime.now().strftime('%F %T')
print("程序结束运行时间：" + end_time)
