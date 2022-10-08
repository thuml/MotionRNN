import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
from core.utils import metrics
from core.utils import preprocess


def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    csi20, csi30, csi40, csi50 = [], [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
            csi20.append(0)
            csi30.append(0)
            csi40.append(0)
            csi50.append(0)

    mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
                csi20[i] += metrics.cal_csi(pred_frm, real_frm, 20)
                csi30[i] += metrics.cal_csi(pred_frm, real_frm, 30)
                csi40[i] += metrics.cal_csi(pred_frm, real_frm, 40)
                csi50[i] += metrics.cal_csi(pred_frm, real_frm, 50)

            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(img_gen_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    if configs.dataset_name == 'echo' or configs.dataset_name == 'guangzhou':
        csi20 = np.asarray(csi20, dtype=np.float32) / batch_id
        csi30 = np.asarray(csi30, dtype=np.float32) / batch_id
        csi40 = np.asarray(csi40, dtype=np.float32) / batch_id
        csi50 = np.asarray(csi50, dtype=np.float32) / batch_id
        print('csi20 per frame: ' + str(np.mean(csi20)))
        for i in range(configs.total_length - configs.input_length):
            print(csi20[i])
        print('csi30 per frame: ' + str(np.mean(csi30)))
        for i in range(configs.total_length - configs.input_length):
            print(csi30[i])
        print('csi40 per frame: ' + str(np.mean(csi40)))
        for i in range(configs.total_length - configs.input_length):
            print(csi40[i])
        print('csi50 per frame: ' + str(np.mean(csi50)))
        for i in range(configs.total_length - configs.input_length):
            print(csi50[i])
