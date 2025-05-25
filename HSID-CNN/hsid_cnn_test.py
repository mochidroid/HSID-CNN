import numpy as np
import scipy.io as sio
import os
import caffe
from tqdm import tqdm

# ----- 設定 -----
caffe.set_mode_gpu()
caffe.set_device(0)

model_def = 'Model/HSID-CNN.prototxt'
# model_weights = 'Model/HSID-CNN_noiselevel100_iter_600000.caffemodel'
model_weights = 'Model/HSID-CNN_noiselevel25.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

k = 12
K = 24

# ----- データの読み込み -----
data = sio.loadmat('Data/GT_crop.mat')
im_label = data['temp']  # shape: (w, h, band)
[w, h, band] = im_label.shape
noiseSigma = 25  # ノイズの標準偏差

# ----- ノイズ付加 -----
im_input = im_label + (noiseSigma / 255.0) * np.random.randn(w, h, band)
# im_input = (im_input - im_input.min()) / (im_input.max() - im_input.min())  # 正規化
im_output = np.zeros_like(im_input)

# print(f"im_label min: {im_label.min():.3f}, max: {im_label.max():.3f}")
# print(f"im_input min: {im_input.min():.3f}, max: {im_input.max():.3f}")

# ----- Caffeに渡すための変換関数 -----
def to_caffe_input(x):
    return x.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

scale_factor = 0.05
residual_cube = np.zeros_like(im_input)

# ----- 処理開始 -----
for i in tqdm(range(band)):
    if i < k:
        spec_cube = im_input[:, :, :K]
    elif i >= band - k:
        spec_cube = im_input[:, :, -K:]
    else:
        spec_cube = im_input[:, :, i-k:i]  # 前半
        spec_cube = np.concatenate((spec_cube, im_input[:, :, i+1:i+k+1]), axis=2)  # 後半

    current_band = im_input[:, :, i]

    net.blobs['data'].data[...] = to_caffe_input(spec_cube)
    net.blobs['data_2'].data[...] = current_band[np.newaxis, np.newaxis, :, :].astype(np.float32)

    net.forward()
    residual = net.blobs['conv10'].data[0, 0, :, :]
    print(f"current_band min: {current_band.min():.3f}, max: {current_band.max():.3f}; residual min: {residual.min():.3f}, max: {residual.max():.3f}")
    # print("Residual band {} stats: min {:.3f}, max {:.3f}, mean {:.3f}".format(
    # i, residual.min(), residual.max(), residual.mean()))
    im_output[:, :, i] = current_band + scale_factor * residual
    residual_cube[:, :, i] = residual

print(f"im_output min: {im_output.min():.3f}, max: {im_output.max():.3f}")

# ----- 保存 -----
sio.savemat('Results/output.mat', {'im_label': im_label, 'im_input': im_input, 'im_output': im_output, 'residual_cube': residual_cube})
