import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import caffe


def make_200x200_from_100x100(cube_100):
    hsi_200 = np.zeros((200, 200, cube_100.shape[2]), dtype=cube_100.dtype)
    for i in range(2):
        for j in range(2):
            hsi_200[i*100:(i+1)*100, j*100:(j+1)*100, :] = cube_100
    return hsi_200


def extract_avg_100x100_from_200x200(cube_200):
    result = np.zeros((100, 100, cube_200.shape[2]), dtype=cube_200.dtype)
    for i in range(2):
        for j in range(2):
            result += cube_200[i*100:(i+1)*100, j*100:(j+1)*100, :]
    return result / 4


def run_hsidcnn_on_file(net, file_path, save_dir):
    data = sio.loadmat(file_path)
    HSI_clean = data['gt']
    HSI_noisy = data['input']

    filename = os.path.basename(file_path).replace('.mat', '')
    padded = make_200x200_from_100x100(HSI_noisy)
    restored_200 = np.zeros_like(padded, dtype=np.float32)

    k = 12
    band = padded.shape[2]

    for i in tqdm(range(band), desc=filename):
        if i < k:
            idx = list(range(0, 24))
        elif i >= band - k:
            idx = list(range(band - 24, band))
        else:
            idx = list(range(i - k, i)) + list(range(i + 1, i + k + 1))

        input_patch = np.transpose(padded[:, :, idx], (2, 0, 1))[np.newaxis, :, :, :]
        target_band = np.transpose(padded[:, :, i], (0, 1))[np.newaxis, np.newaxis, :, :]
        out = net.forward(data=input_patch.astype(np.float32), data_2=target_band.astype(np.float32))
        restored_200[:, :, i] = padded[:, :, i] + out['conv10'][0, 0]

    restored = extract_avg_100x100_from_200x200(restored_200)

    save_path = os.path.join(save_dir, f'{filename}_denoised.mat')
    sio.savemat(save_path, {
        'HSI_clean': HSI_clean,
        'HSI_noisy': HSI_noisy,
        'HSI_restored': restored
    })


def main():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(
        'Model/HSID-CNN.prototxt',
        'Model/HSID-CNN_noiselevel100_iter_600000.caffemodel',
        caffe.TEST
    )

    test_dir = 'Data/JasperRidge'
    save_dir = 'Results_JasperRidge'
    os.makedirs(save_dir, exist_ok=True)

    for case in sorted(os.listdir(test_dir)):
        file_path = os.path.join(test_dir, case, 'data.mat')
        if os.path.exists(file_path):
            run_hsidcnn_on_file(net, file_path, save_dir)


if __name__ == '__main__':
    main()
