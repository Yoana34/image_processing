import os
import cv2
import numpy as np
import torch
from face_detection.retinaface import Retinaface
from face_detection.nets.facenet import Facenet

class FaceRecognition:
    def __init__(self):
        self.retinaface = Retinaface()
        self.face_database_dir = "face_detection/face_dataset"
        if not os.path.exists(self.face_database_dir):
            os.makedirs(self.face_database_dir)
            print(f"创建人脸数据库目录: {self.face_database_dir}")
            
        # 初始化Facenet模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.facenet = Facenet(backbone="mobilenet", mode="predict").to(self.device)
        self.facenet.eval()
        
        # 加载模型权重
        model_path = os.path.join("face_detection/model_data", "facenet_mobilenet.pth")
        if os.path.exists(model_path):
            print(f"加载模型权重: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.facenet.load_state_dict(state_dict, strict=False)
        else:
            print(f"警告: 模型权重文件不存在: {model_path}")
            
        # 预加载数据库中的人脸特征
        self.face_database = {}
        self._load_face_database()
        
    def _normalize_vector(self, v):
        """归一化向量"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _cosine_similarity(self, v1, v2):
        """计算余弦相似度"""
        v1_normalized = self._normalize_vector(v1)
        v2_normalized = self._normalize_vector(v2)
        similarity = np.dot(v1_normalized, v2_normalized)
        # 确保相似度在[-1, 1]范围内
        similarity = np.clip(similarity, -1.0, 1.0)
        return similarity

    def _extract_face_encoding(self, face_image):
        """提取人脸特征编码"""
        try:
            # 预处理
            face_image = cv2.resize(face_image, (160, 160))
            face_image = np.array(face_image, dtype='float32')
            face_image = (face_image - 127.5) / 128.0
            face_image = np.transpose(face_image, (2, 0, 1))
            face_image = np.expand_dims(face_image, axis=0)
            face_tensor = torch.from_numpy(face_image).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                encoding = self.facenet(face_tensor)[0].cpu().numpy()
                encoding = self._normalize_vector(encoding)
            return encoding
        except Exception as e:
            print(f"特征提取错误: {str(e)}")
            return None
        
    def _detect_faces(self, image):
        """检测图像中的人脸"""
        try:
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 检测人脸
            image_with_faces = self.retinaface.detect_image(image_rgb)
            if image_with_faces is None:
                return None, None
            
            # 将结果转换回BGR格式以便OpenCV处理
            image_with_faces = cv2.cvtColor(image_with_faces, cv2.COLOR_RGB2BGR)
            
            # 获取人脸位置
            diff = cv2.absdiff(image, image_with_faces)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            face_locations = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # 过滤掉太小的框
                    face_locations.append([x, y, x+w, y+h])
            
            return face_locations, image_with_faces
        except Exception as e:
            print(f"人脸检测错误: {str(e)}")
            return None, None
        
    def _load_face_database(self):
        """预加载数据库中的所有人脸特征"""
        print("\n开始加载人脸数据库...")
        print(f"数据库目录: {self.face_database_dir}")
        
        if not os.path.exists(self.face_database_dir):
            print("错误: 人脸数据库目录不存在!")
            return
            
        files = os.listdir(self.face_database_dir)
        print(f"发现文件数量: {len(files)}")
        
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                name = os.path.splitext(filename)[0]
                stored_image_path = os.path.join(self.face_database_dir, filename)
                print(f"\n处理文件: {filename}")
                
                try:
                    # 读取图片
                    stored_image = cv2.imdecode(np.fromfile(stored_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if stored_image is None:
                        print(f"错误: 无法读取图片: {filename}")
                        continue
                        
                    print(f"图片尺寸: {stored_image.shape}")
                    
                    # 检测人脸
                    face_locations, _ = self._detect_faces(stored_image)
                    if not face_locations:
                        print(f"错误: 未在数据库图片中检测到人脸: {filename}")
                        continue
                    
                    # 提取第一个人脸
                    x1, y1, x2, y2 = face_locations[0]
                    face_image = stored_image[y1:y2, x1:x2]
                    
                    # 提取特征
                    encoding = self._extract_face_encoding(face_image)
                    if encoding is None:
                        print(f"错误: 无法提取特征: {filename}")
                        continue
                    
                    self.face_database[name] = {
                        'encoding': encoding,
                        'image_path': stored_image_path
                    }
                    print(f"成功加载: {name}")
                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")
                    
        print(f"\n数据库加载完成，共 {len(self.face_database)} 个人脸")
        
    def recognize_face(self, image_path):
        print(f"\n开始识别图片: {image_path}")
        
        # 读取图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("错误: 无法读取输入图像")
            raise ValueError("无法读取图像")
            
        print(f"输入图片尺寸: {image.shape}")
            
        # 检测人脸
        print("开始人脸检测...")
        face_locations, image_with_faces = self._detect_faces(image)
        
        if face_locations is None or not face_locations:
            print("未检测到人脸")
            return None, None, image
            
        print(f"检测到 {len(face_locations)} 个人脸")
        
        # 只处理第一个检测到的人脸
        face = face_locations[0]
        x1, y1, x2, y2 = face
        print(f"人脸位置: ({x1}, {y1}) - ({x2}, {y2})")
        
        # 提取人脸区域
        face_image = image[y1:y2, x1:x2]
        print(f"裁剪的人脸尺寸: {face_image.shape}")
        
        # 提取特征向量
        print("提取特征向量...")
        face_encoding = self._extract_face_encoding(face_image)
        if face_encoding is None:
            print("错误: 无法提取人脸特征")
            return None, None, image
            
        # 在数据库中查找匹配的人脸
        print(f"\n开始在数据库中查找匹配... (数据库大小: {len(self.face_database)})")
        best_match_name = None
        best_match_image = None
        max_similarity = -1
        similarity_threshold = 0.5  # 降低阈值，使匹配更容易
        
        all_similarities = []  # 收集所有相似度用于分析
        for name, data in self.face_database.items():
            stored_encoding = data['encoding']
            similarity = self._cosine_similarity(face_encoding, stored_encoding)
            all_similarities.append((name, similarity))
            print(f"与 {name} 的相似度: {similarity:.4f}")
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = name
                best_match_image = data['image_path']
        
        # 分析所有相似度
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        print("\n所有相似度排序:")
        for name, sim in all_similarities:
            print(f"{name}: {sim:.4f}")
        
        print(f"\n最大相似度: {max_similarity:.4f}, 阈值: {similarity_threshold}")
        # 如果找到匹配的人脸（相似度大于阈值）
        if max_similarity > similarity_threshold:
            print(f"匹配成功: {best_match_name}")
            return best_match_name, best_match_image, image_with_faces
        
        print("未找到匹配的人脸")
        # 如果检测到人脸但没有匹配结果，返回None表示未录入
        return None, None, image_with_faces 