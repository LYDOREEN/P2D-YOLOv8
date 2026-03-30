import warnings
warnings.filterwarnings('ignore')

import torch
import yaml
import cv2
import os
import shutil
import numpy as np
from tqdm import trange
from PIL import Image

from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer,
                 conf_threshold, ratio):

        self.device = torch.device(device)

        ckpt = torch.load(weight, map_location=self.device)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()

        model = Model(cfg, ch=3, nc=len(model_names)).to(self.device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
        model.load_state_dict(csd, strict=False)
        model.eval()

        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        self.model = model
        self.model_names = model_names
        self.conf_threshold = conf_threshold
        self.ratio = ratio

        self.target_layers = [eval(layer)]

        cam_method = eval(method)

        self.cam = cam_method(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=('cuda' in str(self.device))
        )

    def __call__(self, img_path, save_path):

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        img = cv2.imread(img_path)
        img = letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0

        tensor = torch.from_numpy(
            np.transpose(img, (2, 0, 1))
        ).unsqueeze(0).to(self.device)

        # forward
        with torch.no_grad():
            preds = self.model(tensor)[0]

        logits = preds[:, 4:]
        boxes = preds[:, :4]

        scores = logits.max(1)[0]
        sorted_scores, indices = torch.sort(scores, descending=True)

        max_det = max(1, int(len(sorted_scores) * self.ratio))

        for i in trange(max_det):

            if float(sorted_scores[i]) < self.conf_threshold:
                break

            idx = indices[i]

            # 重新forward + backward（GradCAM内部会自动处理）
            grayscale_cam = self.cam(
                input_tensor=tensor,
                targets=None
            )[0]

            cam_image = show_cam_on_image(
                img.copy(),
                grayscale_cam,
                use_rgb=True
            )

            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')


    #model = YOLO('runs/train/yolov8-p2-C2f_ODConv_zengqiang32/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/train/yolov8-p2-C2f_ODConv_zengqiang2/weights/best.pt') # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_v3/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_v2/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2_7.0/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-C2f_ODConv/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8/weights/best.pt')  # 自己训练结束后的模型权重

def get_params():
    params = {
        #'weight': 'runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt',
        #'weight': 'runs/7.0/yolov8-p2-C2f_ODConv_v3/weights/best.pt',
        #'weight': 'runs/7.0/yolov8-p2-C2f_ODConv_v2/weights/best.pt',
        #'weight': 'runs/7.0/yolov8-p2_7.0/weights/best.pt',
        #'weight': 'runs/7.0/yolov8-C2f_ODConv/weights/best.pt',
        'weight': 'runs/7.0/yolov8/weights/best.pt',

        #'cfg': 'yolov8-p2-C2f_ODConv.yaml',
        #'cfg': 'yolov8-p2-C2f_ODConv_v3.yaml',
        #'cfg': 'yolov8-p2-C2f_ODConv_v2.yaml',
        #'cfg': 'yolov8-p2.yaml',
        #'cfg': 'yolov8-C2f_ODConv.yaml',
        'cfg': 'yolov8.yaml',

        'device': 'cuda:0',
        'method': 'HiResCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[18]',
        'conf_threshold': 0.3,  # 0.6
        'ratio': 0.1  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    params = get_params()
    model = yolov8_heatmap(**params)
    model(r'image_heatmap/v1/shiyanqu4_123_27.jpg', 'result2/building/result_a')