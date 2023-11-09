"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
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


ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

DEMO_CH = 'wilk.png'

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


parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples_cartoon')

#### NEW POSE MODEL
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=512, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')

opt_parser = parser.parse_args()

def _2d_vis(points1):
    x, y = points1[:, 0], points1[:, 1]

    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    plt.figure('Draw')

    # 对y轴的坐标系进行反转
    ax = plt.gca()
    ax.invert_yaxis()

    plt.scatter(x, y)  # scatter绘制散点图

    # plt.draw()  # 显示绘图
    plt.show()

def _3d_vis(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2], zdir='z', c='c')
    plt.show()


def vis_2dpoints(img, points):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for p in points:
        cv2.circle(img_rgb, (int(p[0]), int(p[1])), 5, (255, 255, 0), -1)
    plt.imshow(img_rgb)
    plt.show()


# 先得到对应卡通形象的std lmk
DEMO_CH = opt_parser.jpg.split('.')[0]
shape_3d = np.loadtxt('examples_cartoon/{}_face_close_mouth.txt'.format(DEMO_CH))


# python main_end2end_cartoon.py --jpg bluehead.jpg --jpg_bg bluehead_bg.jpg

''' STEP 3: Generate audio data as input to audio branch '''

# 转换好的音频文件统一放到examples文件夹中
c = AutoVC_mel_Convertor('examples',
                         autovc_model_path=opt_parser.load_AUTOVC_name)

# 取得音频文件
au_data = []
au_emb = []
ains = glob.glob1('examples', '*.wav')
ains = [item for item in ains if item is not 'tmp.wav']
ains.sort()
for ain in ains:
    os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))  # 对音频进行重新采样, sr=16000
    # 将转换好的音频保存下来
    shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))
    # au embedding
    wav_path = 'examples/{}'.format(ain)
    me, ae = get_spk_emb(wav_path)  # 使用上面已经转换好的并保存下来的音频
    au_emb.append(me.reshape(-1))  # 得到音频特征emb (其实就是音频编码的均值)
    print('Processing audio file', ain)
    # 将当前的音频的音色统一转换成奥巴马的
    au_data_i = c.convert_single_wav_to_autovc_input(
                                                     audio_filename=wav_path,
                                                     cur_emb=me  # 复用
    )
    au_data += au_data_i
    os.remove(os.path.join('examples', 'tmp.wav'))

# if os.path.isfile('examples/tmp.wav'):
#     os.remove('examples/tmp.wav')

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

if os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
if os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle')):
    os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)  # random_val_au中保存了统一了音色的音频特征
with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
    pickle.dump(gaze, fp)


''' STEP 4: RUN audio->landmark network'''
# 这里上传的shape_3d就是指定的人物的std close mouth lmk, 就并不是使用通用的人脸的std landmark了

image = cv2.imread("examples_cartoon/bluehead.jpg")

# vis_2dpoints(image, (shape_3d[:, :2] + 1) / 2)

# _2d_vis((shape_3d[:, :2] + 1) / 2)


# _2d_vis(shape_3d[:, :2])

model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
if len(opt_parser.reuse_train_emb_list) == 0:
    model.test(au_emb=au_emb)
else:
    model.test(au_emb=None)


# 目前我主要看这个就行, 从这里反推出来针对新的卡通人物要怎么打lmk
''' STEP 5: de-normalize the output to the original image scale '''
fls_names = glob.glob1('examples_cartoon', 'pred_fls_*.txt')
fls_names.sort()
for i in range(0, len(fls_names)):
    ains = glob.glob1('examples', '*.wav')
    ains.sort()
    ain = ains[i]
    fl = np.loadtxt(os.path.join('examples_cartoon', fls_names[i])).reshape((-1, 68, 3))

    # 这里可视化出来的是卡通的关键点
    # _3d_vis(fl[0])

    output_dir = os.path.join('examples_cartoon', fls_names[i][:-4])
    try:
        os.makedirs(output_dir)
    except:
        pass
    # 三角仿射变化要处理拉伸边缘的

    # 这个应该是
    bound, scale, shift = get_puppet_info(DEMO_CH, ROOT_DIR='examples_cartoon')

    fls = fl.reshape((-1, 68, 3))

    fls[:, :, 0:2] = -fls[:, :, 0:2]   # 坐标系转换(其实这里的坐标系转换只是因为scale是负数)
    fls[:, :, 0:2] = fls[:, :, 0:2] / scale  # 尺度还原
    fls[:, :, 0:2] -= shift.reshape(1, 2)  # 平移

    # 这个是与原图是吻合的
    # vis_fl = fls.reshape(-1, 68, 3)[:, :, :2][0]
    # vis_2dpoints(image, vis_fl[0])

    # shape_2d = shape_3d[:, :2]
    # shape_2d = shape_2d / scale
    # shape_2d -= shift.reshape(1, 2)
    # vis_2dpoints(image, shape_2d)

    fls = fls.reshape(-1, 204)

    # additional smooth
    fls[:, 0:48*3] = savgol_filter(fls[:, 0:48*3], 17, 3, axis=0)
    fls[:, 48*3:] = savgol_filter(fls[:, 48*3:], 11, 3, axis=0)
    fls = fls.reshape((-1, 68, 3))

    # 对关键点进行一定的修改, 以适应三角形仿射变换的拉伸
    if DEMO_CH in ['paint', 'mulaney', 'cartoonM', 'beer', 'color', 'JohnMulaney', 'vangogh', 'jm', 'roy', 'lineface']:
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
    static_frame = np.loadtxt(os.path.join('examples_cartoon', '{}_face_open_mouth.txt'.format(DEMO_CH)))
    # vis_2dpoints(image, static_frame)
    # assert False
    static_frame_2d = static_frame[r, 0:2]
    static_frame_2d = np.concatenate((static_frame_2d, bound.reshape(-1, 2)), axis=0)
    np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame_2d, fmt='%.2f')

    # triangle_vtx_index.txt
    shutil.copy(os.path.join('examples_cartoon', DEMO_CH + '_delauney_tri.txt'),
                os.path.join(output_dir, 'triangulation.txt'))

    os.remove(os.path.join('examples_cartoon', fls_names[i]))

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
    if(os.name == 'nt'): 
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
    os.system('ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i {} -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -shortest -strict -2 {}'.format(
        os.path.join(cur_dir, '..', '..', '..', 'examples', ain),
        os.path.join(cur_dir, '..', 'out.mp4')
    ))
