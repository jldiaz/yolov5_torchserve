from inspect import trace
import time
import traceback

from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import torch
import torchvision
import cv2


# Taken from yolov5/utils/augmentations.py
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.batch_size = 1
        self.img_size = 640

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        list_img_names = ["img" + str(i) for i in range(1, self.batch_size + 1)]
        inputs = torch.zeros(self.batch_size, 3, self.img_size, self.img_size)
        for i, img_name in enumerate(list_img_names):
            try:
                # Take the input data and make it inference ready
                byte_array = data[0][img_name]
                file_bytes = np.asarray(bytearray(byte_array), dtype=np.uint8)
                # yolov5 preprocessing
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                # THIS WAS ADDED. Otherwise the API only accepts 640x640 images
                # The following line rescales the image so that the result is always 640x640
                # The stride is made equal to the image widht/height, so that filling bands are always added
                img = letterbox(img, (self.img_size, self.img_size), stride=self.img_size)[0]  # padded resize
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                input = torch.from_numpy(img)
                input = input.float()
                input /= 255.0  # 0 - 255 to 0.0 - 1.0
                inputs[i, :, :, :] = input
            except Exception as e:
                print(traceback.format_exc())
        return inputs

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        pred = non_max_suppression(postprocess_output[0], conf_thres=0.6)
        pred = [p.tolist() for p in pred]
        return [pred]


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.6, classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Number of classes.
    nc = prediction[0].shape[1] - 5

    # Candidates.
    xc = prediction[..., 4] > conf_thres

    # Settings:
    # Minimum and maximum box width and height in pixels.
    min_wh, max_wh = 2, 256

    # Maximum number of detections per image.
    max_det = 100

    # Timeout.
    time_limit = 10.0

    # Require redundant detections.
    redundant = True

    # Multiple labels per box (adds 0.5ms/img).
    multi_label = nc > 1

    # Use Merge-NMS.
    merge = False

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply constraints:
        # Confidence.
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling.
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image.
        if not x.shape[0]:
            continue

        # Compute conf.
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2).
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls).
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:

            # Best class only.
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class.
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image.
        # Number of boxes.
        n = x.shape[0]
        if not n:
            continue

        # Batched NMS:
        # Classes.
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS.
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        # Limit detections.
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):

            # Merge NMS (boxes merged using weighted mean).
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4).
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
