import numpy as np
import cv2
import mediapipe as mp
from skimage import transform as trans
import matplotlib.pyplot as plt



def vis_2d_points(points1, points2):

    x, y = points1[:, 0], -points1[:, 1]

    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    plt.figure('Draw')

    plt.scatter(x, y)  # scatter绘制散点图

    # plt.draw()  # 显示绘图

    x, y = points2[:, 0], -points2[:, 1]

    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多

    plt.scatter(x, y)  # scatter绘制散点图


    plt.show()


class MpDetection:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)

    face_det = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=0
    )

    def norm_crop(self, img, lmk, image_size=112, shrink_factor=1.0):
        """
        image_size: 320
        """
        arcface_src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()

        """
        这是一个scale对齐的因子, 这个因子会先将arcface_src缩放到image_size
        """
        src_factor = image_size / 112

        """
        shrink_factor是一个缩小因子, 缩小之后还需要将点向右下角平移, 给视野的外扩留出空间
        """
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56

        # vis_2d_points(arcface_src, src)


        """
        if shrink_factor == 1.0
            lmk就会对齐到scale=112的空间上
        elif shrink_factor < 1.0
            这个情况就是arcface_src已经被缩小了, 但是src_factor是按照112的标准计算的, 所以src是缩小了的, 将lmk对齐到缩小之后的src之后, 其实是没有占到image_size这个scale大小的, 是比image_size小的
            但是在函数warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)中的(image_size, image_size)还是不变的
            所以就产生了对齐+外扩的效果
        
        """

        src = src * src_factor  # 如果shrink_factor是1的话, lmk和src是一样的, 就是等scale的对齐, 如果src的scale变大了, lmk除了对齐之外就还有外扩了
        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(img, M, (image_size, image_size),
                                borderMode=cv2.BORDER_REPLICATE,
                                # borderValue=0.0
                                )
        return warped, tform.params

    def face_aligne(self, image, image_size, shrink_factor):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ld_index = [468, 473, 1, 61, 291]
        img_h, img_w, _ = img.shape

        # 人脸框检测
        results_det = self.face_det.process(img)
        if results_det.detections is None:
            return None, None, None

        # 选择面积最大的bbox
        max_index = 0
        max_are = 0
        for i in range(len(results_det.detections)):
            bb_infor = results_det.detections[i].location_data.relative_bounding_box
            tmp_are = bb_infor.height * bb_infor.width
            if tmp_are > max_are:
                max_are = tmp_are
                max_index = i
        bb = results_det.detections[max_index].location_data.relative_bounding_box

        img_bb = img[int(np.clip(bb.ymin, 0, 1) * img_h):int((np.clip(bb.ymin, 0, 1) + bb.height) * img_h),
                 int(np.clip(bb.xmin, 0, 1) * img_w):int((np.clip(bb.xmin, 0, 1) + bb.width) * img_w)]
        bb_h, bb_w, _ = img_bb.shape

        # 关键点检测
        results = self.face_mesh.process(img_bb)
        if results.multi_face_landmarks is None:
            return None, None, None

        ld_img = []
        for ld in ld_index:
            tmp_xyz = results.multi_face_landmarks[0].landmark[ld]
            ld_img.append([tmp_xyz.x * bb_w + bb.xmin * img_w, tmp_xyz.y * bb_h + bb.ymin * img_h])
        img_lm = np.array(ld_img)  # 其实这里我得到lm就可以了

        img_aligned, M = self.norm_crop(image, img_lm, image_size=image_size, shrink_factor=shrink_factor)
        return img_aligned, img_lm, M


    def face_aligne_w_lmk(self, image, image_size, shrink_factor, img_lm):
        img_aligned, M = self.norm_crop(image, img_lm, image_size=image_size, shrink_factor=shrink_factor)
        return img_aligned, img_lm, M


    def process(self, image, image_size=512, shrink_factor=0.5, lmk=None):
        if lmk is not None:
            return self.face_aligne_w_lmk(image, image_size, shrink_factor, lmk)  # bgr
        return self.face_aligne(image, image_size, shrink_factor)  # bgr

if __name__ == '__main__':
    detec = MpDetection()

    image_path = "D:/novelai/fzy/lora-scripts/dataset/head_only/shijiaqi_meitu_nobody/shijiaqi_0001.jpg"

    name = 'cl1'
    img = cv2.imread(image_path)
    aligned, _, _ = detec.process(img, shrink_factor=0.75)
    plt.imshow(aligned)
    plt.show()
