from scipy.io import loadmat, savemat
import numpy as np
import caffe
import os
import matplotlib.pyplot as plt
import metrics

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

# モデル読み込み
net = caffe.Net("Model/HSID-CNN.prototxt", "Model/HSID-CNN_noiselevel100_iter_600000.caffemodel", caffe.TEST)

# 入力データ
mat = loadmat('Data/GT_crop.mat')  # shape: (h, w, bands)
im_label = mat['temp'].astype(np.float32)
w, h, bands = im_label.shape

# ノイズ付与
noiseSigma = 100.0
im_input = im_label + (noiseSigma / 255.0) * np.random.randn(*im_label.shape).astype(np.float32)

# 出力初期化
denoised_cube = np.zeros((w, h, bands), dtype=np.float32)

# 推論処理（1バンドずつ）
for i in range(bands):
    if i < 12 or i > bands - 13:
        # 端のバンドは元のノイズ付きバンドをそのまま使用
        denoised_cube[:, :, i] = im_input[:, :, i]
        continue

    input_patch = im_input[:, :, [j for j in range(i - 12, i)] + [j for j in range(i + 1, i + 13)]]
    target_band = im_input[:, :, i]

    # reshape to (1, 24, h, w) and (1, 1, h, w)
    input_patch = np.transpose(input_patch, (2, 0, 1))[np.newaxis, :, :, :]
    target_band = target_band[np.newaxis, np.newaxis, :, :]

    # 推論実行
    output = net.forward(data=input_patch, data_2=target_band)
    result = output['conv10'][0, 0] + target_band[0, 0]
    denoised_cube[:, :, i] = result

    print(f'Band {i+1}/{bands} done')

# # 保存（.mat形式）
# savemat('result_denoised_cube.mat', {'denoised_cube': denoised_cube})
# print("全バンドの復元結果を result_denoised_cube.mat に保存しました。")


# --- Evaluate ---
mpsnr, mssim = metrics.cal_metrics(im_label, im_input)

print(f'MPSNR: {mpsnr:.2f}, MSSIM: {mssim:.4f}')