import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util.mp_face_aligned import MpDetection
from util.get_bluehead_delauney_tri import BlueheadDelauneyTri



import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CartoonPreprocess(object):
    def __init__(self):
        self.mp_align = MpDetection()
        self.bdt = BlueheadDelauneyTri()

    def run(self, save_dir="examples_cartoon", DEMO_CH = "womanteacher"):
        lmk_txt = "{}_open_mouth.txt".format(DEMO_CH)
        image = cv2.imread("{}_open_mouth.png".format(DEMO_CH))
        lines = open(lmk_txt, "r").readlines()
        points_np = self.get_points(lines)
        # self.vis_2dpoints(image, points_np)

        aligned_image, aligned_lmk = self.image_lmk_uniform(self.mp_align, image, points_np)
        aligned_lmk = np.array(aligned_lmk, dtype=int)

        # 将对齐之后的open mouth的img和lmk保存下来
        cv2.imwrite(os.path.join(save_dir, "{}.jpg".format(str(DEMO_CH))), aligned_image)
        np.savetxt(os.path.join(save_dir, "{}_face_open_mouth.txt".format(DEMO_CH)), aligned_lmk)

        # 使用norm_anno对得到的open mouth的lmk进行处理,得到对应的norm的close mouth
        self.norm_anno(save_dir, DEMO_CH, param=[0.7, 0.4, 0.5, 0.5], show=True)

        # 形成三角形仿射变换的各个切面
        img_path = os.path.join(save_dir, "{}.jpg".format(DEMO_CH))
        lmk_path = os.path.join(save_dir, "{}_face_open_mouth.txt".format(DEMO_CH))
        self.bdt.run(img_path, lmk_path)

    def vis_2dpoints(self, img, points):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 255, 0), -1)
        plt.imshow(img)
        plt.show()

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

    def get_points(self, lines):
        points_np = []
        for line in lines:
            point_num = line.strip().split("points=")[-1]
            point_num = point_num[1:-2]
            point_num = point_num.split(",")
            points_np.append([float(point_num[0]), float(point_num[1])])
        points_np = np.array(points_np)
        return points_np


    def image_lmk_uniform(self, mp_align:MpDetection, image, lmk):
        five_lmk = []

        eye_left = (lmk[37] + lmk[38] + lmk[40] + lmk[41]) / 4
        eye_right = (lmk[43] + lmk[44] + lmk[46] + lmk[47]) / 4
        nose = lmk[30]
        mouth_left = lmk[48]
        mouth_right = lmk[54]

        five_lmk.append(eye_left)
        five_lmk.append(eye_right)
        five_lmk.append(nose)
        five_lmk.append(mouth_left)
        five_lmk.append(mouth_right)

        five_lmk = np.array(five_lmk)

        aligned_image, _, M = mp_align.process(image, lmk=five_lmk, shrink_factor=0.9)

        # 构建齐次式
        one = np.ones((len(lmk), 1))
        aligned_lmk = np.concatenate([lmk, one], axis=-1)
        aligned_lmk = np.matmul(M, aligned_lmk.transpose()).transpose()

        # self.vis_2dpoints(aligned_image, aligned_lmk)

        return aligned_image, aligned_lmk

    def standardization(self, aligned_image, aligned_lmk):
        h, w = aligned_image.shape[:2]

        aligned_lmk[:, 0] -= (w / 2)
        aligned_lmk[:, 1] -= (h / 2)

        aligned_lmk[:, 0] = aligned_lmk[:, 0] * (1 / w)
        aligned_lmk[:, 1] = aligned_lmk[:, 1] * (1 / h)

        scale = ((1 / w) + (1 / h)) / 2

        shift_x = - w / 2
        shift_y = - h / 2

        return scale, shift_x, shift_y

    def destandardization(self, aligned_image, aligned_norm_lmk, scale, shift_x, shift_y):
        aligned_norm_lmk /= scale
        aligned_norm_lmk[:, 0] -= shift_x
        aligned_norm_lmk[:, 1] -= shift_y
        # self.vis_2dpoints(aligned_image, aligned_norm_lmk)

    def norm_anno(self, ROOT_DIR, CH, param=[0.75, 0.35, 0.6, 0.6], show=True):
        face_tmp = np.loadtxt(os.path.join(ROOT_DIR, CH + '_face_open_mouth.txt'))  # .reshape(1, 204)
        face_tmp = face_tmp.reshape(68, 3)

        scale = 1.6 / (face_tmp[0, 0] - face_tmp[16, 0])
        shift = - 0.5 * (face_tmp[0, 0:2] + face_tmp[16, 0:2])
        face_tmp[:, 0:2] = (face_tmp[:, 0:2] + shift) * scale
        face_std = np.loadtxt(os.path.join(ROOT_DIR, 'STD_FACE_LANDMARKS.txt'))
        face_std = face_std.reshape(68, 3)

        face_tmp[:, -1] = face_std[:, -1]
        face_tmp[:, 0:2] = -face_tmp[:, 0:2]
        np.savetxt(os.path.join(ROOT_DIR, CH + '_face_open_mouth_norm.txt'), face_tmp, fmt='%.4f')
        np.savetxt(os.path.join(ROOT_DIR, CH + '_scale_shift.txt'), np.array([scale, shift[0], shift[1]]), fmt='%.10f')

        # Force the frame to close mouth
        face_tmp[49:54, 1] = param[0] * face_tmp[49:54, 1] + (1-param[0]) * face_tmp[59:54:-1, 1]
        face_tmp[59:54:-1, 1] = param[1] * face_tmp[49:54, 1] + (1-param[1]) * face_tmp[59:54:-1, 1]
        face_tmp[61:64, 1] = param[2] * face_tmp[61:64, 1] + (1-param[2]) * face_tmp[67:64:-1, 1]
        face_tmp[67:64:-1, 1] = param[3] * face_tmp[61:64, 1] + (1-param[3]) * face_tmp[67:64:-1, 1]
        face_tmp[61:64, 0] = 0.6 * face_tmp[61:64, 0] + 0.4 * face_tmp[67:64:-1, 0]
        face_tmp[67:64:-1, 0] = 0.6 * face_tmp[61:64, 0] + 0.4 * face_tmp[67:64:-1, 0]

        np.savetxt(os.path.join(ROOT_DIR, CH + '_face_close_mouth.txt'), face_tmp, fmt='%.4f')

        std_face_id = np.loadtxt(os.path.join(ROOT_DIR, CH + '_face_close_mouth.txt'))  # .reshape(1, 204)
        std_face_id = std_face_id.reshape(68, 3)

        def vis_landmark_on_plt(fl, x_offset=0.0, show_now=True):
            def draw_curve(shape, idx_list, loop=False, x_offset=0.0, c=None):
                for i in idx_list:
                    plt.plot((shape[i, 0] + x_offset, shape[i + 1, 0] + x_offset), (-shape[i, 1], -shape[i + 1, 1]), c=c)
                if (loop):
                    plt.plot((shape[idx_list[0], 0] + x_offset, shape[idx_list[-1] + 1, 0] + x_offset),
                             (-shape[idx_list[0], 1], -shape[idx_list[-1] + 1, 1]), c=c)

            draw_curve(fl, list(range(0, 16)), x_offset=x_offset)  # jaw
            draw_curve(fl, list(range(17, 21)), x_offset=x_offset)  # eye brow
            draw_curve(fl, list(range(22, 26)), x_offset=x_offset)
            draw_curve(fl, list(range(27, 35)), x_offset=x_offset)  # nose
            draw_curve(fl, list(range(36, 41)), loop=True, x_offset=x_offset)  # eyes
            draw_curve(fl, list(range(42, 47)), loop=True, x_offset=x_offset)
            draw_curve(fl, list(range(48, 59)), loop=True, x_offset=x_offset, c='b')  # mouth
            draw_curve(fl, list(range(60, 67)), loop=True, x_offset=x_offset, c='r')
            draw_curve(fl, list(range(60, 64)), loop=False, x_offset=x_offset, c='g')

            if show_now:
                plt.show()

        vis_landmark_on_plt(std_face_id, show_now=show)


if __name__ == "__main__":
    """
    这里是处理张嘴代码的, 闭嘴的关键点也是通过张嘴的生成的
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='anne.jpg', help='the space name of processing img')
    opt_parser = parser.parse_args()

    cp = CartoonPreprocess()
    DEMO_CH = opt_parser.img
    cp.run(DEMO_CH=DEMO_CH)










