# 数字图像处理系统

这是一个基于Python和Flask开发的数字图像处理系统，提供了丰富的图像处理功能。

## 功能实现
### 基本功能：
- 图像运算
- 图像增强
- 图像分割
- 图像平滑
- 图像锐化
- 数学形态学
- 图像恢复

![image1](https://github.com/Yoana34/image_processing/blob/main/images/0436d8aaf492ed7ba2e902cf6c81d07.png)
### 场景功能：车管所
- **人脸识别：通过输入图像，输出已存储人脸的名字及照片。**

![image2](https://github.com/Yoana34/image_processing/blob/main/images/296cc6ddcdd0b2339faa01f69c546e2a_.png)
![image3](https://github.com/Yoana34/image_processing/blob/main/images/c478e6c7683c273d80127dd2cbc55d1f_.png)
- **图像去雾：能够对雾天等状况下有雾监控照片进行去雾**

![image4](https://github.com/Yoana34/image_processing/blob/main/images/72d9e30edc1f2089669839a0eb35f41.png)
## 环境配置

### 安装步骤
1.克隆项目并创建虚拟环境
```bash
git clone https://github.com/Yoana34/image_processing.git
```

2.安装依赖包
* Python 
* PyTorch >= 1.2.0
* torchvision
* numpy
* skimage
* h5py
* opencv-python
* matplotlib
* typing
* scipy
* tqdm
* Pillow

（需要配备pytorch+cuda）

3.安装模型
将要用到的模型[预训练模型](https://drive.google.com/open?id=1da13IOlJ3FQfH6Duj_u1exmZzgXPaYXe)下载到defogging/models路径下。

4.运行项目
```bash
python app.py
```

### 说明

1. 启动应用后，访问 `http://localhost:5000` 打开Web界面
2. 选择需要处理的图片
3. 选择相应的处理功能
4. 调整参数（如果需要）
5. 点击处理按钮获取结果

## 项目结构

```
digital_process/
├── app.py                    # Flask应用主文件，包含所有Web路由和API接口
├── processing.py             # 图像处理基础功能实现
├── face_recognition.py       # 人脸识别模块
├── defogging_process.py      # 图像去雾模块
├── static/                   # 静态资源文件目录
│   ├── css/                 # 样式文件
│   └── js/                  # JavaScript文件
├── templates/               # HTML模板目录
│   └── index.html          # 主页面模板
├── uploads/                # 图片存储
├── face_detection/         # 人脸检测相关内容
└── defogging/             # 图像去雾相关内容

```

## 注意事项
1. 上传图片大小限制为16MB
2. 支持的图片格式：PNG、JPG、JPEG、BMP
3. 处理大图片时可能需要较长时间，请耐心等待

## 项目引用
https://github.com/bubbliiiing/facenet-retinaface-pytorch

https://github.com/BookerDeWitt/MSBDN-DFF
