from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import uuid
from processing import *
from face_recognition import FaceRecognition
import cv2
from defogging_process import Defogging

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 初始化人脸识别
face_recognition = FaceRecognition()
defogging = Defogging()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return path, filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 图像加法、减法、乘法
@app.route('/api/arithmetic', methods=['POST'])
def api_arithmetic():
    try:
        op_type = request.form.get('op_type')
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        if not (file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': '请上传两张图片'}), 400
        path1, _ = save_file(file1)
        path2, _ = save_file(file2)
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        image_arithmetic(op_type, path1, out_path, path2)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 图像几何变换
@app.route('/api/geometric', methods=['POST'])
def api_geometric():
    try:
        transform_type = request.form.get('transform_type')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        geometric_transform(path, out_path, transform_type, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 仿射变换
@app.route('/api/affine', methods=['POST'])
def api_affine():
    try:
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        # src_pts, dst_pts 以json字符串传递
        import json
        src_pts = json.loads(request.form.get('src_pts'))
        dst_pts = json.loads(request.form.get('dst_pts'))
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        affine_transform(path, out_path, src_pts, dst_pts)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 傅里叶变换
@app.route('/api/fourier', methods=['POST'])
def api_fourier():
    try:
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        fourier_transform(path, out_path, 0)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 图像增强
@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    try:
        method = request.form.get('method')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        enhance_image(path, out_path, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 形态学
@app.route('/api/morphology', methods=['POST'])
def api_morphology():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_morphology(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 噪声模拟
@app.route('/api/noise', methods=['POST'])
def api_noise():
    try:
        noise_type = request.form.get('noise_type')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        if noise_type == 'gaussian':
            mean = float(request.form.get('mean', 0))
            var = float(request.form.get('var', 0.1))
            add_gaussian_noise(path, out_path, mean, var)
        elif noise_type == 'salt_pepper':
            salt_prob = float(request.form.get('salt_prob', 0.05))
            pepper_prob = float(request.form.get('pepper_prob', 0.05))
            add_salt_pepper_noise(path, out_path, salt_prob, pepper_prob)
        else:
            return jsonify({'error': '未知噪声类型'}), 400
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 均值滤波
@app.route('/api/filter/mean', methods=['POST'])
def api_filter_mean():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        apply_image_filters(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 排序滤波
@app.route('/api/filter/sort', methods=['POST'])
def api_filter_sort():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        apply_image_filters(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 选择滤波
@app.route('/api/filter/select', methods=['POST'])
def api_filter_select():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        apply_image_filters(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# 边缘检测
@app.route('/api/edge', methods=['POST'])
def api_edge():
    try:
        method = request.form.get('method')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        edge_detection(path, out_path, method)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 线性变化检测（霍夫变换）
@app.route('/api/line', methods=['POST'])
def api_line():
    try:
        method = request.form.get('method')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        line_detection(path, out_path, method, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
"""
# 空域/频域平滑
@app.route('/api/smooth', methods=['POST'])
def api_smooth():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_smoothing(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
"""
# 空域平滑
@app.route('/api/smooth/space', methods=['POST'])
def api_smooth_space():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_smoothing(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 频域平滑
@app.route('/api/smooth/frequency', methods=['POST'])
def api_smooth_frequency():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_smoothing(path, out_path,operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




"""
# 空域/频域锐化
@app.route('/api/sharpen', methods=['POST'])
def api_sharpen():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_sharpening(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
"""
# 空域锐化
@app.route('/api/sharpen/space', methods=['POST'])
def api_sharpen_space():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_sharpening(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 频域锐化
@app.route('/api/sharpen/frequency', methods=['POST'])
def api_sharpen_frequency():
    try:
        operation = request.form.get('operation')
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        path, _ = save_file(file)
        params = request.form.to_dict()
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        process_sharpening(path, out_path, operation, params)
        return jsonify({'result': f'/uploads/{out_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 人脸识别
@app.route('/api/face/recognition', methods=['POST'])
def api_face_recognition():
    try:
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
        
        path, _ = save_file(file)
        name, stored_image, processed_image = face_recognition.recognize_face(path)
        
        # 保存处理后的图像
        output_filename = f"processed_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, processed_image)
        
        # 如果有匹配的人脸图片，复制到上传目录
        matched_face_filename = None
        if stored_image:
            matched_face_filename = f"matched_{uuid.uuid4().hex}.png"
            matched_face_path = os.path.join(app.config['UPLOAD_FOLDER'], matched_face_filename)
            # 复制文件
            import shutil
            shutil.copy2(stored_image, matched_face_path)
        
        result = {
            'result': f'/uploads/{output_filename}',
            'matched_name': name,
            'matched_face': f'/uploads/{matched_face_filename}' if matched_face_filename else None
        }
        
        print("API返回结果:", result)  # 调试信息
        return jsonify(result)
    except Exception as e:
        print("API错误:", str(e))  # 调试信息
        return jsonify({'error': str(e)}), 500

#图像去雾
@app.route('/api/defogging', methods=['POST'])
def api_defogging():
    try:
        file = request.files.get('file')
        if not (file and allowed_file(file.filename)):
            return jsonify({'error': '请上传图片'}), 400
            
        path, _ = save_file(file)
        out_name = f"processed_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        
        # 进行去雾处理
        if not defogging.process_image(path, out_path):
            return jsonify({'error': '去雾处理失败，请确保模型文件存在且格式正确'}), 500
            
        # 检查输出文件是否成功生成
        if not os.path.exists(out_path):
            return jsonify({'error': '去雾处理失败，无法生成输出文件'}), 500
            
        return jsonify({'result': f'/uploads/{out_name}'})
        
    except Exception as e:
        print("去雾处理错误:", str(e))
        return jsonify({'error': f'去雾处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 