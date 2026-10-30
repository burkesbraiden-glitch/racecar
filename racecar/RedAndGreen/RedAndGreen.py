# 导入代码依赖
import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import torch
import IPython.display
from ais_bench.infer.interface import InferSession
from det_utils import letterbox, scale_coords, nms

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LightPublisher(Node):
    def __init__(self):
        super().__init__('light_publisher')
        self.publisher_ = self.create_publisher(String, '/light', 10)
        self.img_publisher_ = self.create_publisher(Image, '/camera/image_result', 10)
        self.bridge = CvBridge()

def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]#将BGR颜色空间转换为RGB颜色空间。
    img = img.transpose(2, 0, 1)  # HWC2CHW
    img = np.ascontiguousarray(img, dtype=np.float16) / 255.0
    return img, scale_ratio, pad_size#调整后的图片，以及缩放比例和填充大小。
global dete_result 
dete_result = ''
def draw_bbox(bbox, img0, color, wt, names):
    """在图片上画预测框"""
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        global dete_result
        dete_result = names[int(class_id)]
        print(names[int(class_id)])
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    cv2.imwrite('test.png',img0)#将绘制好的图像保存为test.png文件
    return img0


def get_labels_from_txt(path):
    """从txt文件获取图片标签"""
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict

# pred（预测框信息）
# image（原始图片）
# image（原始图片）
def draw_prediction(pred, image, labels):
    """在图片上画出预测框并进行可视化展示"""
    imgbox = widgets.Image(format='jpg', height=720, width=1280)
    img_dw = draw_bbox(pred, image, (0, 255, 0), 2, labels)
    imgbox.value = cv2.imencode('.jpg', img_dw)[1].tobytes()
    display(imgbox)


def infer_image(img_path, model, class_names, cfg):
    """图片推理"""
    # 图片载入
    image = cv2.imread(img_path)
    if image is None:
        print(f"找不到图片: {img_path}，请检查路径是否正确！")
        return

    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    
    # 判断是否检测到目标
    if len(pred_all) == 0:
        print("未检测到任何目标。")
        return

    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    
    # 【修改这里】：直接调用 draw_bbox 进行画框、打印和保存
    print("----- 识别结果 -----")
    draw_bbox(pred_all, image, (0, 255, 0), 2, class_names)
    print("--------------------")
    print("推理完成！画好框的图片已保存为当前目录下的 test.png")


def infer_frame_with_vis(image, model, labels_dict, cfg, bgr2rgb=True):
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    # 模型推理
    output = model.infer([img])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    # 图片预测结果可视化
    img_vis = draw_bbox(pred_all, image, (0, 255, 0), 2, labels_dict)
    return img_vis


def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode('.jpg', image)[1])


def infer_video(video_path, model, labels_dict, cfg, output_path='output.mp4'):
    """视频推理"""
    image_widget = widgets.Image(format='jpeg', width=800, height=600)
    display(image_widget)

    # 读入视频
    cap = cv2.VideoCapture(video_path)
    video_writer = None
    
    while cap.isOpened():
        ret, img_frame = cap.read()
        if not ret:
            break
            
        # 对视频帧进行推理
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg, bgr2rgb=False)
        image_widget.value = img2bytes(image_pred[:, :, ::-1])

        if video_writer is None:
            # 初始化视频写入对象
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
        # 将每一帧的预测结果写入视频对象
        video_writer.write(image_pred)
        
    cap.release()
    if video_writer is not None:
        video_writer.release()


def infer_camera(model, labels_dict, cfg):
    """外设摄像头实时推理"""
    def find_camera_index(): #查找可用的摄像头索引
        max_index_to_check = 10  # 扩大搜索范围到10，防止外接相机编号靠后
        for index in range(max_index_to_check):
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened() and cap.read()[0]:
                cap.release()
                return index
        # If no camera is found
        raise ValueError("No camera found.")

    rclpy.init()
    light_node = LightPublisher()

    # 获取摄像头
    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 调高获取的画面分辨率以提升rqt端清晰度，原640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 调高获取的画面分辨率以提升rqt端清晰度，原480
    print("画面正在发布到 ROS 2 话题: /camera/image_result")
    print("请在另一个终端使用 rqt_image_view 查看画面。按 Ctrl+C 退出程序。")

    while rclpy.ok():
        # 获取摄像头画面
        ret, img_frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 摄像头安装方向为倒置，旋转180°
        img_frame = cv2.rotate(img_frame, cv2.ROTATE_180)
        # 进行模型推理和画预测框
        image_pred = infer_frame_with_vis(img_frame, model, labels_dict, cfg)
        
        # 【新增：将带有预测框的画面发布到 ROS 2 话题】
        # 将 OpenCV 的 BGR 格式数组转换成 ROS 的 Image 消息
        img_msg = light_node.bridge.cv2_to_imgmsg(image_pred, encoding="bgr8")
        light_node.img_publisher_.publish(img_msg)

        global dete_result
        if dete_result:
            msg = String()
            msg.data = dete_result
            light_node.publisher_.publish(msg)
            dete_result = None
            
        # 【关键：让 ROS 2 处理底层事件，确保消息顺利发出去】
        rclpy.spin_once(light_node, timeout_sec=0.01)

    # 清理并释放资源
    cap.release()
    light_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    cfg = {
        'conf_thres': 0.45,  # 模型置信度阈值，阈值越低，得到的预测框越多(原0.7，降为0.25提升远距离小目标检测率)
        'iou_thres': 0.40,  # IOU阈值，高于这个阈值的重叠预测框会被过滤掉
        'input_shape': [640, 640],  # 模型输入尺寸
    }

    model_path = 'yolov5s_bs1_light.om'
    label_path = './coco_names.txt'
    # 初始化推理模型
    model = InferSession(0, model_path)
    labels_dict = get_labels_from_txt(label_path)
    infer_mode = 'camera'
    # infer_mode = 'image'  # 可选 'image', 'camera', 'video'

    if infer_mode == 'image':
        img_path = '0010.jpg'
        infer_image(img_path, model, labels_dict, cfg)
    elif infer_mode == 'camera':
        infer_camera(model, labels_dict, cfg)
    elif infer_mode == 'video':
        video_path = '0.mp4'
        infer_video(video_path, model, labels_dict, cfg, output_path='output_test.mp4')

    
    
    
