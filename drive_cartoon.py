"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import glob
import numpy as np
import argparse
import pickle
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
from src.approaches.train_audio2landmark import Audio2landmark_model
import shutil
import matplotlib.pyplot as plt
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
from scipy.signal import savgol_filter
from util.utils import get_puppet_info
import cv2
import os
import torch
from src.dataset.audio2landmark.audio2landmark_dataset import Audio2landmark_Dataset

ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True
# examples_cartoon/bluehead.jpg
parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, required=True, help='Puppet image name to animate (with filename extension), e.g. wilk.png')
parser.add_argument('--jpg_bg', type=str, required=True, help='Puppet image background (with filename extension), e.g. wilk_bg.jpg')
parser.add_argument('--out', type=str, default='out.mp4')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth') #ckpt_audio2landmark_g.pth') #
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_i2i_finetune_150.pth') #ckpt_image2image.pth') #

parser.add_argument('--amp_lip_x', type=float, default=2.0)
parser.add_argument('--amp_lip_y', type=float, default=2.0)
parser.add_argument('--amp_pos', type=float, default=0.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['E_kmpT-EfOg']) #  ['E_kmpT-EfOg']) # ['45hn7-LXDX8'])

parser.add_argument('--output_folder', type=str, default='examples_cartoon')

opt_parser = parser.parse_args()

# python drive_cartoon.py --jpg womanteacher.jpg --jpg_bg womanteacher_bg.jpg

class Drive_Cartoon():
    def __init__(self, data_dir="examples"):
        self.data_dir = data_dir
        self.c = AutoVC_mel_Convertor(data_dir, autovc_model_path=opt_parser.load_AUTOVC_name)
        self.DEMO_CH = opt_parser.jpg.split('.')[0]
        self.shape_3d = np.loadtxt('examples_cartoon/{}_face_close_mouth.txt'.format(self.DEMO_CH))
        # self.shape_3d = np.loadtxt('examples_cartoon/{}_face_close_mouth.txt'.format("bluehead"))
        self.model = Audio2landmark_model(opt_parser,
                                          jpg_shape=self.shape_3d,
                                          build_data=False,  # 先不创建数据集, 等__run__()中处理完audio数据之后再创建数据集
                                          load_embs=False,
                                          data_dir=self.data_dir
                                          )

    def __run__(self):
        # au_data是奥巴马的声音特征，au_emb是原声音的嵌入特征
        au_data, au_emb = self.process_audio()
        # 建立数据缓存
        self.buffle_creation(au_data)

        # 处理好缓存数据
        # 读取奥巴马的脸部点特征和原声换奥巴声音特征
        self.model.eval_data = Audio2landmark_Dataset(dump_dir='{}/dump'.format(self.data_dir),
                                                      dump_name='random',
                                                      status='val',
                                                      num_window_frames=18,
                                                      num_window_step=1)
        self.model.eval_dataloader = torch.utils.data.DataLoader(self.model.eval_data,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 collate_fn=self.model.eval_data.my_collate_in_segments)

        fls_dict = self.audio2landmark(au_emb=au_emb)
        self.lmk_denormal_and_facewarp(fls_dict=fls_dict)

    def process_audio(self):
        """
        self.data_dir: 其中可以存储多个wav文件
        每个文件生成的结果都会放在self.au_emb和self.au_data中
        """
        au_data = []
        au_emb = []

        ains = glob.glob1(self.data_dir, '*.wav')
        ains = [item for item in ains if item is not 'tmp.wav']
        ains.sort()
        for ain in ains:
            # 将输入文件的音频采样率更改为 16,000 Hz 并保存到指定的输出文件中
            os.system('ffmpeg -y -loglevel error -i {}/{} -ar 16000 {}/tmp.wav'.format(self.data_dir, ain, self.data_dir))
            # 将转换好的音频保存下来
            shutil.copyfile('{}/tmp.wav'.format(self.data_dir), '{}/{}'.format(self.data_dir, ain))
            # au embedding
            wav_path = '{}/{}'.format(self.data_dir, ain)
            # me是全部声纹嵌入的平均嵌入总值再平均，ae是全部声纹嵌入的平均嵌入总值
            me, ae = get_spk_emb(wav_path)  # 使用上面已经转换好的并保存下来的音频
            au_emb.append(me.reshape(-1))  # 得到音频特征emb (其实就是音频编码的均值)
            print('Processing audio file', ain)
            # 将当前的音频的音色统一转换成奥巴马的
            au_data_i = self.c.convert_single_wav_to_autovc_input(
                audio_filename=wav_path,
                cur_emb=me  # 复用
            )
            au_data += au_data_i
            # self.au_data.append(au_data_i)  # au_data_i本来就是个元组, 但是使用append的话就会在每个item后面加上一层[], 就莫名其妙了

            os.remove(os.path.join('{}'.format(self.data_dir), 'tmp.wav'))

        return au_data, au_emb

    def buffle_creation(self, au_data):
        fl_data = []
        rot_tran, rot_quat, anchor_t_shape = [], [], []
        for au, info in au_data:
            """
            info: 其中包含了au(音色统一了的音频), wav文件名, wav中的音色emb
            """
            au_length = au.shape[0]
            # 以下都只是缓冲buffle
            fl_data.append((np.zeros(shape=(au_length, 68 * 3)), info))
            rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
            rot_quat.append(np.zeros(shape=(au_length, 4)))
            anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

        if os.path.exists(os.path.join(self.data_dir, 'dump', 'random_val_fl.pickle')):
            os.remove(os.path.join(self.data_dir, 'dump', 'random_val_fl.pickle'))
        if os.path.exists(os.path.join(self.data_dir, 'dump', 'random_val_fl_interp.pickle')):
            os.remove(os.path.join(self.data_dir, 'dump', 'random_val_fl_interp.pickle'))
        if os.path.exists(os.path.join(self.data_dir, 'dump', 'random_val_au.pickle')):
            os.remove(os.path.join(self.data_dir, 'dump', 'random_val_au.pickle'))
        if os.path.exists(os.path.join(self.data_dir, 'dump', 'random_val_gaze.pickle')):
            os.remove(os.path.join(self.data_dir, 'dump', 'random_val_gaze.pickle'))

        with open(os.path.join(self.data_dir, 'dump', 'random_val_fl.pickle'), 'wb') as fp:
            pickle.dump(fl_data, fp)
        with open(os.path.join(self.data_dir, 'dump', 'random_val_au.pickle'), 'wb') as fp:
            pickle.dump(au_data, fp)  # random_val_au中保存了统一了音色的音频特征
        with open(os.path.join(self.data_dir, 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
            gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
            pickle.dump(gaze, fp)

    def audio2landmark(self, au_emb):
        if len(opt_parser.reuse_train_emb_list) == 0:
            fls_dict = self.model.test(au_emb=au_emb, is_return_fls_dict=True)
        else:
            fls_dict = self.model.test(au_emb=None, is_return_fls_dict=True)
        return fls_dict

    def lmk_denormal_and_facewarp(self, fls_dict=None):
        # 读取之前生成好的关键点
        if fls_dict is not None:
            fls_names = list(fls_dict.keys()).sort()
        else:
            fls_names = glob.glob1('examples_cartoon', 'pred_fls_*.txt')
            fls_names.sort()

        ains = glob.glob1(self.data_dir, '*.wav')
        ains.sort()

        for i in range(0, len(fls_names)):
            ain = ains[i]
            if fls_dict is not None:
                fl = fls_dict[fls_names[i]].reshape((-1, 68, 3))
            else:
                fl = np.loadtxt(os.path.join('examples_cartoon', fls_names[i])).reshape((-1, 68, 3))

            # 这里可视化出来的是卡通的关键点
            # self._3d_vis(fl[0])

            output_dir = os.path.join('examples_cartoon', fls_names[i][:-4]) if fls_dict is None else os.path.join('examples_cartoon', fls_names[i])
            try:
                os.makedirs(output_dir)
            except:
                pass
            # 三角仿射变化要处理拉伸边缘的

            # 这个应该是
            bound, scale, shift = get_puppet_info(self.DEMO_CH, ROOT_DIR='examples_cartoon')

            fls = fl.reshape((-1, 68, 3))

            # for cur_fls in fls:
            #     self._2d_vis(cur_fls)

            fls[:, :, 0:2] = -fls[:, :, 0:2]  # 坐标系转换(其实这里的坐标系转换只是因为scale是负数)
            fls[:, :, 0:2] = fls[:, :, 0:2] / scale  # 尺度还原
            fls[:, :, 0:2] -= shift.reshape(1, 2)  # 平移

            # 可视化预测出来的点
            # vis_fl = fls.reshape(-1, 68, 3)[:, :, :2]
            # image = np.array(np.zeros((512, 512, 3)), dtype=np.uint8)
            # for cur_vis_fl in vis_fl:
            #     self.vis_2dpoints(image, cur_vis_fl)

            # 可视化std 2d
            # shape_2d = -self.shape_3d[:, :2]
            # shape_2d = shape_2d / scale
            # shape_2d -= shift.reshape(1, 2)
            # image = np.array(np.zeros((512, 512, 3)), dtype=np.uint8)
            # self.vis_2dpoints(image, shape_2d)

            fls = fls.reshape(-1, 204)

            # additional smooth
            fls[:, 0:48 * 3] = savgol_filter(fls[:, 0:48 * 3], 17, 3, axis=0)
            fls[:, 48 * 3:] = savgol_filter(fls[:, 48 * 3:], 11, 3, axis=0)
            fls = fls.reshape((-1, 68, 3))

            # 对关键点进行一定的修改, 以适应三角形仿射变换的拉伸
            if self.DEMO_CH in ['paint',
                                'mulaney',
                                'cartoonM',
                                'beer',
                                'color',
                                'JohnMulaney',
                                'vangogh',
                                'jm',
                                'roy',
                                'lineface']:
                r = list(range(0, 68))
                fls = fls[:, r, :]
                fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
                fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
                fls = fls.reshape(-1, 160)

            else:
                r = list(range(0, 48)) + list(range(60, 68))
                # print(fls.shape)  # (287, 68, 3)
                fls = fls[:, r, :]
                # print(fls.shape)  # (287, 56, 3)
                fls = fls[:, :, 0:2].reshape(-1, 56 * 2)
                # print(fls.shape)  # (287, 112)
                # for fl in fls:
                #     _2d_vis(fl.reshape(56, 2))
                # 使用bound替换掉fls缺少的部分
                fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
                # print(fls.shape)  # (287, 136)
                fls = fls.reshape(-1, 112 + bound.shape[1])
                # print(fls.shape)  # (287, 136)
                # print("----------------------------------------------------")
            # 最后将得到的关键点存入到warped_points.txt供后续使用
            np.savetxt(os.path.join(output_dir, 'warped_points.txt'), fls, fmt='%.2f')

            # 使用张嘴的点作为一个参考
            # static_points.txt
            # 这个是张嘴的点, 这是个非常直观的点, 与图片对应没有进行任何缩放和平移
            static_frame = np.loadtxt(os.path.join('examples_cartoon', '{}_face_open_mouth.txt'.format(self.DEMO_CH)))

            # image = np.array(np.zeros((512, 512, 3)), dtype=np.uint8)
            # self.vis_2dpoints(image, static_frame)
            # assert False
            static_frame_2d = static_frame[r, 0:2]
            static_frame_2d = np.concatenate((static_frame_2d, bound.reshape(-1, 2)), axis=0)
            np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame_2d, fmt='%.2f')

            # triangle_vtx_index.txt
            shutil.copy(os.path.join('examples_cartoon', self.DEMO_CH + '_delauney_tri.txt'),
                        os.path.join(output_dir, 'triangulation.txt'))

            os.remove(os.path.join('examples_cartoon', fls_names[i])) if fls_dict is None else os.path.join('examples_cartoon', fls_names[i] + ".txt")

            self.face_warp(output_dir, ain)


    def face_warp(self, output_dir, ain):
        # 根据关键点生成视频
        # ==============================================
        # Step 4 : Vector art morphing
        # ==============================================
        warp_exe = os.path.join(os.getcwd(), 'facewarp', 'facewarp.exe')
        if os.path.exists(os.path.join(output_dir, 'output')):
            shutil.rmtree(os.path.join(output_dir, 'output'))
        os.mkdir(os.path.join(output_dir, 'output'))
        os.chdir('{}'.format(os.path.join(output_dir, 'output')))
        cur_dir = os.getcwd()
        print(cur_dir)
        if os.name == 'nt':
            ''' windows '''
            os.system('{} {} {} {} {} {}'.format(
                warp_exe,
                os.path.join(cur_dir, '..', '..', opt_parser.jpg),
                os.path.join(cur_dir, '..', 'triangulation.txt'),
                os.path.join(cur_dir, '..', 'reference_points.txt'),
                os.path.join(cur_dir, '..', 'warped_points.txt'),
                os.path.join(cur_dir, '..', '..', opt_parser.jpg_bg),
                '-novsync -dump'))
        else:
            ''' linux '''
            os.system('wine {} {} {} {} {} {}'.format(
                warp_exe,
                os.path.join(cur_dir, '..', '..', opt_parser.jpg),
                os.path.join(cur_dir, '..', 'triangulation.txt'),
                os.path.join(cur_dir, '..', 'reference_points.txt'),
                os.path.join(cur_dir, '..', 'warped_points.txt'),
                os.path.join(cur_dir, '..', '..', opt_parser.jpg_bg),
                '-novsync -dump'))
        os.system(
            # 拼接上语音
            'ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i {} -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -shortest -strict -2 {}'.format(
                os.path.join(cur_dir, '..', '..', '..', self.data_dir, ain),
                os.path.join(cur_dir, '..', 'out.mp4')
            ))


    def _2d_vis(self, points):
        x, y = points[:, 0], points[:, 1]

        # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
        plt.figure('Draw')

        # 对y轴的坐标系进行反转
        ax = plt.gca()
        ax.invert_yaxis()

        plt.scatter(x, y)  # scatter绘制散点图

        # plt.draw()  # 显示绘图
        plt.show()

    def _3d_vis(self, points):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2], zdir='z', c='c')
        plt.show()


    def vis_2dpoints(self, img, points):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for p in points:
            cv2.circle(img_rgb, (int(p[0]), int(p[1])), 5, (255, 255, 0), -1)
        plt.imshow(img_rgb)
        plt.show()


if __name__ == "__main__":
    drive_cartoon = Drive_Cartoon()
    drive_cartoon.__run__()