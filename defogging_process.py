import torch
import numpy as np
from PIL import Image
import os
import cv2
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
defogging_dir = os.path.join(current_dir, "defogging")

# 添加所有必要的路径
paths_to_add = [parent_dir, defogging_dir]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)
        print(f"Added path: {path}")

try:
    from defogging.networks import MSBDN_RDFF
except ImportError as e:
    print(f"Error importing MSBDN_RDFF module: {str(e)}")
    print(f"Python path: {sys.path}")
    raise

class Defogging:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = None
        
    def load_model(self):
        if self.model is None:
            model_path = os.path.join(defogging_dir, "models", "model.pkl")
            try:
                print(f"Loading model from: {model_path}")
                # 直接加载整个模型
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return True
    
    def preprocess_image(self, img):
        """预处理图像以确保尺寸正确"""
        # 将图像调整为固定大小（根据模型的要求）
        target_size = (1024, 1024)  # 使用1024x1024作为标准尺寸
        img = img.resize(target_size, Image.LANCZOS)
        
        # 确保尺寸是8的倍数（因为模型中有3次下采样）
        w, h = img.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        if w != img.size[0] or h != img.size[1]:
            img = img.resize((w, h), Image.LANCZOS)
            
        return img
        
    def process_image(self, input_path, output_path):
        try:
            # 加载模型
            if not self.load_model():
                return False
                
            # 读取并预处理图像
            img = Image.open(input_path).convert('RGB')
            original_size = img.size  # 保存原始尺寸
            img = self.preprocess_image(img)  # 添加预处理步骤
            img = np.array(img) / 255.0  # 归一化到[0,1]
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            img = img.unsqueeze(0).to(self.device)
            
            # 打印输入张量的形状以便调试
            print(f"Input tensor shape: {img.shape}")
            
            # 进行去雾处理
            with torch.no_grad():
                output = self.model(img)
                output = torch.clamp(output, 0, 1)
                
                # 转换回numpy数组
                output = output[0].cpu().numpy()
                output = np.transpose(output, (1, 2, 0))
                output = (output * 255).astype(np.uint8)
                
                # 将输出调整回原始尺寸
                output = cv2.resize(output, (original_size[0], original_size[1]), interpolation=cv2.INTER_LANCZOS4)
                
                # 保存结果
                cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
                print(f"Defogged image saved to: {output_path}")
                return True
                
        except Exception as e:
            print(f"Error in defogging process: {str(e)}")
            print(f"Error details:", sys.exc_info())  # 添加更详细的错误信息
            return False 