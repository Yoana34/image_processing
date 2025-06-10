import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import math
from defogging_process import Defogging

# 初始化去雾处理器
defogging = Defogging()

def enhance_image(input_path, output_path, params):
    """
    Apply custom enhancement operations to an image.

    Parameters:
    -----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the processed image.
    params : dict
        Dictionary of parameters for the enhancement:
        - method: 'hist_eq', 'hist_norm', 'log', 'linear', 'defogging'
        - alpha, beta: float, used in linear transform (optional)
    """
    method = params.get('method', '')

    if method == 'hist_eq':
        # 直方图均衡化
        img = cv2.imread(input_path)
        if len(img.shape) == 3:
            # 彩色图像，对每个通道分别进行直方图均衡化
            b, g, r = cv2.split(img)
            b = cv2.equalizeHist(b)
            g = cv2.equalizeHist(g)
            r = cv2.equalizeHist(r)
            img = cv2.merge([b, g, r])
        else:
            # 灰度图像
            img = cv2.equalizeHist(img)
        cv2.imwrite(output_path, img)
        
    elif method == 'hist_norm':
        # 直方图正规化
        img = cv2.imread(input_path)
        if len(img.shape) == 3:
            # 彩色图像，对每个通道分别进行直方图正规化
            b, g, r = cv2.split(img)
            b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
            r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.merge([b, g, r])
        else:
            # 灰度图像
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(output_path, img)
        
    elif method == 'log':
        # 对数变换
        img = cv2.imread(input_path)
        img = np.float32(img)
        img = cv2.log(1.0 + img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)
        cv2.imwrite(output_path, img)
        
    elif method == 'linear':
        # 线性变换
        alpha = float(params.get('alpha', 1.0))
        beta = float(params.get('beta', 0))
        img = cv2.imread(input_path)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite(output_path, img)
        
    elif method == 'defogging':
        # 图像去雾
        if not defogging.process_image(input_path, output_path):
            raise Exception("去雾处理失败")
            
    else:
        raise ValueError(f"不支持的增强方法: {method}")

#数学形态学
def process_morphology(input_path, output_path, operation, params):
    img = cv2.imread(input_path)

    # 灰度图像
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 获取结构元参数
    kernel_shape = params.get('kernel_shape', 'rect')  #类型
    kernel_size = int(params.get('kernel_size', 5))   #大小

    # 结构元
    if kernel_shape == 'rect': #矩形结构元
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == 'ellipse': #椭圆结构元
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'cross':  #交叉结构元
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:  #默认值为1的正方形结构元
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 二值化处理
    if 'threshold' in params and params['threshold']:
        threshold = int(params['threshold'])
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        binary = gray

    #形态学操作
    if operation == 'erosion':  #腐蚀
        iterations = int(params.get('iterations', 1)) if params.get('iterations') else 1
        result = cv2.erode(binary, kernel, iterations=iterations)

    elif operation == 'dilation': #膨胀
        iterations = int(params.get('iterations', 1)) if params.get('iterations') else 1
        result = cv2.dilate(binary, kernel, iterations=iterations)

    elif operation == 'opening': #开运算
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    elif operation == 'closing': #闭运算
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


    if len(img.shape) > 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, result)
    return output_path

#添加高斯噪声
def add_gaussian_noise(input_path: str, output_path: str, mean: float = 0, var: float = 0.1) -> str:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_path}")

    # 归一化图像
    normalized = image.astype(float) / 255.0
    # 生成高斯噪声
    noise = np.random.normal(mean, var, normalized.shape)
    # 叠加噪声并裁剪到[0,1]范围
    noisy = np.clip(normalized + noise, 0.0, 1.0)
    # 恢复为0-255范围
    result = (noisy * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    return output_path

#添加椒盐噪声
def add_salt_pepper_noise(input_path: str, output_path: str, salt_prob: float = 0.05, pepper_prob: float = 0.05) -> str:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_path}")

    result = image.copy()
    # 椒噪声
    pepper = np.random.rand(*image.shape[:2]) < pepper_prob
    result[pepper] = 0
    # 盐噪声
    salt = np.random.rand(*image.shape[:2]) < salt_prob
    if len(image.shape) > 2:  # 彩色图像
        result[salt] = 255
    else:  # 灰度图像
        result[salt] = 255

    cv2.imwrite(output_path, result)
    return output_path

# 图像恢复（滤波）
def apply_image_filters(input_path: str, output_path: str, operation: str, params: Optional[Dict] = None) -> str:
    if params is None:
        params = {}

    # 读取输入图像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_path}")

    # 均值滤波优化
    if operation in ['arithmetic_mean', 'geometric_mean', 'harmonic_mean']:
        kernel_size = int(params.get('kernel_size', 3))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        if operation == 'arithmetic_mean':  # 算术均值滤波
            # 使用OpenCV的boxFilter实现算术均值滤波
            result = cv2.boxFilter(image, -1, (kernel_size, kernel_size), normalize=True)
        else:
            # 对于几何均值和谐波均值，使用滑动窗口视图优化
            pad_size = kernel_size // 2
            padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
            view = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))

            if operation == 'geometric_mean':  # 几何均值滤波向量化实现
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_view = np.log(view.astype(np.float32) + 1e-6)
                    result = np.exp(np.mean(log_view, axis=(2, 3)))
            else:  # 谐波均值滤波向量化实现
                with np.errstate(divide='ignore', invalid='ignore'):
                    reciprocal_view = 1.0 / (view.astype(np.float32) + 1e-6)
                    sum_reciprocal = np.sum(reciprocal_view, axis=(2, 3))
                    result = (kernel_size * kernel_size) / sum_reciprocal

            result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
            result = np.clip(result, 0, 255).astype(np.uint8)

    # 排序滤波优化
    elif operation in ['max_filter', 'min_filter', 'median_filter']:
        kernel_size = int(params.get('kernel_size', 3))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        if operation == 'max_filter':
            result = cv2.dilate(image, np.ones((kernel_size, kernel_size)))
        elif operation == 'min_filter':
            result = cv2.erode(image, np.ones((kernel_size, kernel_size)))
        elif operation == 'median_filter':
            result = cv2.medianBlur(image, kernel_size)

    # 选择滤波优化
    elif operation in ['low_pass', 'high_pass', 'band_pass', 'band_stop']:
        if operation == 'low_pass':
            threshold = np.uint8(params.get('threshold', 128))
            result = np.where(image < threshold, image, 0).astype(np.uint8)
        elif operation == 'high_pass':
            threshold = np.uint8(params.get('threshold', 128))
            result = np.where(image > threshold, image, 0).astype(np.uint8)
        elif operation == 'band_pass':
            min_val = np.uint8(params.get('min_val', 50))
            max_val = np.uint8(params.get('max_val', 200))
            result = np.where((image >= min_val) & (image <= max_val), image, 0).astype(np.uint8)
        elif operation == 'band_stop':
            min_val = np.uint8(params.get('min_val', 50))
            max_val = np.uint8(params.get('max_val', 200))
            result = np.where((image < min_val) | (image > max_val), image, 0).astype(np.uint8)

    else:
        raise ValueError(f"未知的操作类型: {operation}")

    cv2.imwrite(output_path, result)
    return output_path

#边缘检测
def edge_detection(input_path, output_path, method='sobel'):
    """
    图像边缘检测
    method: 'roberts', 'prewitt', 'sobel', 'laplacian', 'log', 'canny'
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {input_path}")

    if method == 'roberts':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        kernelx = np.array([[-1, 0], [0, 1]], dtype=np.int32)
        kernely = np.array([[0, -1], [1, 0]], dtype=np.int32)

        x = cv2.filter2D(gray_img, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray_img, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    elif method == 'prewitt':
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        x = cv2.filter2D(img, -1, kernelx)
        y = cv2.filter2D(img, -1, kernely)
        result = cv2.convertScaleAbs(cv2.addWeighted(x, 0.5, y, 0.5, 0))

    elif method == 'sobel':
        x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(x, y)
        result = cv2.convertScaleAbs(result)

    elif method == 'laplacian':
        lap = cv2.Laplacian(img, cv2.CV_64F)
        result = cv2.convertScaleAbs(lap)

    elif method == 'log':
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        result = cv2.convertScaleAbs(lap)

    elif method == 'canny':
        result = cv2.Canny(img, 100, 200)

    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    cv2.imwrite(output_path, result)

#线性变化
def line_detection(input_path, output_path, method='hough', params=None):
    """
    线性变化检测（霍夫变换）
    method: 'hough'（标准），'houghp'（概率）
    params: 可选参数，如阈值、最小线长等
    """
    if params is None:
        params = {}

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Unable to read image at {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if method == 'hough':
        lines = cv2.HoughLines(edges, 1, np.pi / 180, int(params.get('threshold', 150)))
        result = img.copy()
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    elif method == 'houghp':
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=int(params.get('threshold', 100)),
                                minLineLength=np.double(params.get('min_line_length', 50)),
                                maxLineGap=np.double(params.get('max_line_gap', 10)))
        result = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        raise ValueError(f"Unsupported Hough method: {method}")

    cv2.imwrite(output_path, result)

#图像锐化
def process_sharpening(input_path: str, output_path: str, operation: str, params: Dict = None) -> str:
    if params is None:
        params = {}

    # 读取输入图像并转为灰度
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_path}")

    # 空域锐化 - 使用OpenCV优化函数和向量化操作
    if operation in ['roberts', 'sobel', 'prewitt', 'laplacian']:
        if operation == 'roberts':
            # Roberts算子优化实现
            roberts_x = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            roberts_y = np.array([[1, 0], [0, -1]], dtype=np.float32)

            # 使用OpenCV的filter2D进行卷积
            gx = cv2.filter2D(image, cv2.CV_32F, roberts_x)
            gy = cv2.filter2D(image, cv2.CV_32F, roberts_y)

            # 计算梯度幅值
            magnitude = cv2.magnitude(gx, gy)
            result = np.clip(magnitude, 0, 255).astype(np.uint8)

        elif operation == 'sobel':
            # 使用OpenCV的Sobel函数
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

            # 计算梯度幅值
            magnitude = cv2.magnitude(gx, gy)
            result = np.clip(magnitude, 0, 255).astype(np.uint8)

        elif operation == 'prewitt':
            # Prewitt算子核
            prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

            # 使用filter2D进行卷积
            gx = cv2.filter2D(image, cv2.CV_32F, prewitt_x)
            gy = cv2.filter2D(image, cv2.CV_32F, prewitt_y)

            # 计算梯度幅值
            magnitude = cv2.magnitude(gx, gy)
            result = np.clip(magnitude, 0, 255).astype(np.uint8)

        elif operation == 'laplacian':
            # 使用OpenCV的Laplacian函数
            result = cv2.Laplacian(image, cv2.CV_16S)
            result = cv2.convertScaleAbs(result)

        cv2.imwrite(output_path, result)

    # 频域锐化优化
    elif operation in ['ideal_high', 'butterworth_high', 'gaussian_high']:
        # 获取参数
        D0 = float(params.get('D0', 40))  # 默认截止频率40
        n = float(params.get('n', 2))  # 默认阶数2

        # 傅里叶变换
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # 使用NumPy的meshgrid创建距离矩阵
        y, x = np.ogrid[:rows, :cols]
        y_dist = y - crow
        x_dist = x - ccol
        D = np.sqrt(x_dist ** 2 + y_dist ** 2)

        # 创建滤波器模板 (向量化操作)
        if operation == 'ideal_high':
            filter_mask = (D >= D0).astype(np.float32)
        elif operation == 'butterworth_high':
            with np.errstate(divide='ignore', invalid='ignore'):
                filter_mask = 1 / (1 + (D0 / np.where(D == 0, 1e-10, D)) ** (2 * n))
                filter_mask[D == 0] = 0
        elif operation == 'gaussian_high':
            filter_mask = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

        # 频域滤波处理
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * filter_mask
        idft_shift = np.fft.ifftshift(filtered_dft)
        filtered_img = np.abs(np.fft.ifft2(idft_shift))

        # 归一化并转换为8位图像
        result = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_path, result)

    else:
        raise ValueError(f"未知的锐化类型: {operation}")

    return output_path

#图像平滑
def process_smoothing(input_path: str, output_path: str, operation: str, params: Dict = None) -> str:
    if params is None:
        params = {}

    # 读取输入图像
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_path}")

    # 空域平滑 - 使用OpenCV优化函数
    if operation == 'median_3x3':  # 3x3中值滤波
        result = cv2.medianBlur(image, 3)
    elif operation == 'median_5x5':  # 5x5中值滤波
        result = cv2.medianBlur(image, 5)
    elif operation == 'mean':  # 邻域平均法
        result = cv2.blur(image, (3, 3))

    # 频域平滑优化
    elif operation in ['ideal_low', 'butterworth_low', 'gaussian_low']:
        D0 = float(params.get('D0', 20))
        n = float(params.get('n', 2))

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # 使用NumPy的meshgrid替代循环
        y, x = np.ogrid[:rows, :cols]
        y_dist = y - crow
        x_dist = x - ccol
        D = np.sqrt(x_dist ** 2 + y_dist ** 2)

        if operation == 'ideal_low':  # 理想低通滤波
            filter_mask = (D <= D0).astype(np.float32)
        elif operation == 'butterworth_low':  # 巴特沃斯低通滤波
            with np.errstate(divide='ignore', invalid='ignore'):
                filter_mask = 1 / (1 + 0.414 * (D / D0) ** (2 * n))
                filter_mask[np.isinf(filter_mask)] = 1  # 处理D=0的情况
        elif operation == 'gaussian_low':  # 高斯低通滤波
            filter_mask = np.exp(-(D ** 2) / (2 * (D0 ** 2)))

        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * filter_mask
        idft_shift = np.fft.ifftshift(filtered_dft)
        filtered_img = np.abs(np.fft.ifft2(idft_shift))
        result = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)


    cv2.imwrite(output_path, result)
    return output_path


def read_images(img1_path, img2_path=None):
    img1 = cv2.imread(img1_path)
    if img1 is None:
        raise ValueError(f"Unable to read image at {img1_path}")
    if img2_path:
        img2 = cv2.imread(img2_path)
        if img2 is None:
            raise ValueError(f"Unable to read image at {img2_path}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2
    return img1, None

#图像算术运算
def image_arithmetic(op_type, input1, output, input2=None):
    """
    图像加法、减法、乘法
    op_type: 'add', 'subtract', 'multiply'
    """
    img1, img2 = read_images(input1, input2)

    if op_type == 'add':
        result = cv2.add(img1, img2)
    elif op_type == 'subtract':
        result = cv2.subtract(img1, img2)
    elif op_type == 'multiply':
        result = cv2.multiply(img1, img2)
    else:
        raise ValueError(f"Unsupported operation type: {op_type}")

    cv2.imwrite(output, result)

#图像几何变换
def geometric_transform(input_path, output_path, transform_type, params):
    """
    图像几何变换：缩放、平移、旋转、翻转
    transform_type: 'scale', 'translate', 'rotate', 'flip'
    params: dict with transformation parameters
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Unable to read image at {input_path}")

    rows, cols = img.shape[:2]

    if transform_type == 'scale':
        fx = float(params.get('fx', 1.0))  # 强制转换，避免字符串或 None
        fy = float(params.get('fy', 1.0))
        result = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    elif transform_type == 'translate':
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(img, M, (cols, rows))

    elif transform_type == 'rotate':
        try:
            angle_str = params.get('angle', '0')  # 先作为字符串获取
            scale_str = params.get('scale', '1.0')  # 先作为字符串获取
            print(f"Raw angle value: {params.get('angle')}, type: {type(params.get('angle'))}")
            print(f"Raw scale value: {params.get('scale')}, type: {type(params.get('scale'))}")
            angle = float(angle_str)
            scale = float(scale_str)
        except (TypeError, ValueError):
            raise ValueError("angle and scale must be numbers")
        center = (cols // 2, rows // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(img, M, (cols, rows))

    elif transform_type == 'flip':
        flip_code =int( params.get('code', 1) ) # 0: vertical, 1: horizontal, -1: both
        result = cv2.flip(img, flip_code)

    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")

    cv2.imwrite(output_path, result)

#仿射变换
def affine_transform(input_path, output_path, src_pts, dst_pts):
    """
    仿射变换
    src_pts 和 dst_pts 应为 [(x1,y1), (x2,y2), (x3,y3)] 格式的三个点列表
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Unable to read image at {input_path}")

    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    rows, cols = img.shape[:2]
    result = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(output_path, result)

#傅里叶变换
def fourier_transform(input_path, output_path,a):
    """
    仅执行傅里叶变换（DFT），生成频域幅度谱图像
    """
    # 读取图像（灰度模式）
    img = cv2.imread(input_path, 0)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")

    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 将零频率分量移到频谱中心
    dft_shift = np.fft.fftshift(dft)

    # 计算幅度谱（log缩放增强视觉效果）
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log(magnitude + 1)  # +1避免log(0)

    # 归一化到0-255并保存
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, np.uint8(magnitude_normalized))


