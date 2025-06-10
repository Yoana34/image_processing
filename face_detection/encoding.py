import os
import cv2
import numpy as np
import torch
from .retinaface import Retinaface
from .nets.facenet import Facenet

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''

# 确保face_dataset目录存在
dataset_dir = os.path.join(os.path.dirname(__file__), "face_dataset")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

retinaface = Retinaface(1)

# 初始化Facenet模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = Facenet(backbone="mobilenet", mode="predict").to(device)
facenet.eval()

# 如果目录为空，跳过编码过程
list_dir = os.listdir(dataset_dir)
if list_dir:
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append(os.path.join(dataset_dir, name))
        names.append(name.split("_")[0])

    retinaface.encode_face_dataset(image_paths,names)

def encode_face(image, face_location):
    """
    提取人脸特征向量
    :param image: 原始图像
    :param face_location: 人脸位置 [x1, y1, x2, y2]
    :return: 128维特征向量
    """
    try:
        x1, y1, x2, y2 = face_location[:4]
        face_image = image[y1:y2, x1:x2]
        
        # 调整大小为160x160（Facenet要求的输入大小）
        face_image = cv2.resize(face_image, (160, 160))
        
        # 预处理图像
        face_image = np.array(face_image, dtype='float32')
        face_image = np.expand_dims(face_image.transpose(2, 0, 1), 0)
        face_image = (face_image - 127.5) / 128.0
        
        # 转换为PyTorch张量并移动到正确的设备
        face_image = torch.from_numpy(face_image).to(device)
            
        # 使用Facenet提取特征
        with torch.no_grad():
            face_encoding = facenet(face_image)[0].cpu().numpy()
            # 归一化特征向量
            face_encoding = face_encoding / np.linalg.norm(face_encoding)
        return face_encoding
        
    except Exception as e:
        print(f"提取人脸特征时出错: {str(e)}")
        return None

def compare_faces(face_encoding1, face_encoding2):
    """
    比较两个人脸特征向量的相似度
    :param face_encoding1: 第一个人脸特征向量
    :param face_encoding2: 第二个人脸特征向量
    :return: 欧氏距离（值越小表示越相似）
    """
    try:
        # 计算欧氏距离
        distance = np.linalg.norm(face_encoding1 - face_encoding2)
        return distance
        
    except Exception as e:
        print(f"比较人脸特征时出错: {str(e)}")
        return float('inf')  # 返回无穷大表示比较失败
