import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def cal_mpsnr(gt, output):
    """
    Calculate Mean PSNR (MPSNR) between ground truth and output.
    
    Parameters:
    gt (numpy.ndarray): Ground truth image.
    output (numpy.ndarray): Output image.
    
    Returns:
    float: Mean PSNR value.
    """
    psnr_values = []
    for i in range(gt.shape[2]):
        psnr_value = compare_psnr(gt[:, :, i], output[:, :, i], data_range=1.0)
        psnr_values.append(psnr_value)
    return np.mean(psnr_values)


def cal_mssim(gt, output):
    """
    Calculate Mean SSIM (MSSIM) between ground truth and output.
    
    Parameters:
    gt (numpy.ndarray): Ground truth image.
    output (numpy.ndarray): Output image.
    
    Returns:
    float: Mean SSIM value.
    """
    ssim_values = []
    for i in range(gt.shape[2]):
        ssim_value = compare_ssim(gt[:, :, i], output[:, :, i], data_range=1.0)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

def cal_metrics(gt, output):
    """
    Calculate PSNR and SSIM metrics between ground truth and output.
    
    Parameters:
    gt (numpy.ndarray): Ground truth image.
    output (numpy.ndarray): Output image.
    
    Returns:
    tuple: Mean PSNR and Mean SSIM values.
    """
    mpsnr = cal_mpsnr(gt, output)
    mssim = cal_mssim(gt, output)
    return mpsnr, mssim
