// 分类与功能映射
const subCategoryMap = {
    arithmetic: [
        { value: "add", label: "图像加法" },
        { value: "subtract", label: "图像减法" },
        { value: "multiply", label: "图像乘法" },
        { value: "geometric", label: "扩展缩放/平移/旋转/翻转" },
        { value: "affine", label: "仿射变换" },
        { value: "fourier", label: "傅里叶变换" }
    ],
    enhance: [
        { value: "hist_eq", label: "直方图均衡化" },
        { value: "log", label: "对数变换" },
        { value: "linear", label: "线性变换" }
    ],
    segment: [
        { value: "edge", label: "边缘检测" },
        { value: "line", label: "线性变化检测" }
    ],
    smooth: [
        { value: "space", label: "空域的平滑" },
        { value: "frequency", label: "频域的平滑" }
    ],
    sharpen: [
        { value: "space", label: "空域的锐化" },
        { value: "frequency", label: "频域的锐化" }
    ],
    morphology: [
        { value: "opening", label: "开运算" },
        { value: "closing", label: "闭运算" },
        { value: "erosion", label: "腐蚀" },
        { value: "dilation", label: "膨胀" }
    ],
    restore: [
        { value: "noise", label: "噪声模拟" },
        { value: "mean", label: "均值滤波" },
        { value: "sort", label: "排序滤波" },
        { value: "select", label: "选择滤波" }
    ],
    face: [
        { value: "recognition", label: "人脸识别" }
    ],
    defogging: [
        { value: "defogging", label: "图像去雾" }
    ]
};

// 二级功能的详细选项
const detailOptions = {
    edge: [
        { value: "roberts", label: "Roberts" },
        { value: "prewitt", label: "Prewitt" },
        { value: "sobel", label: "Sobel" },
        { value: "laplacian", label: "Laplacian" },
        { value: "log", label: "LoG" },
        { value: "canny", label: "Canny" }
    ],
    line: [
        { value: "hough", label: "HoughLines" },
        { value: "houghp", label: "HoughLinesP" }
    ],
    space: [
        { value: "median_3x3", label: "中值滤波法（3x3）" },
        { value: "median_5x5", label: "中值滤波法（5x5）" },
        { value: "mean", label: "邻域平均法" }
    ],
    frequency: [
        { value: "ideal_low", label: "低通滤波" },
        { value: "butterworth_low" , label: "巴特沃斯低通滤波" },
        { value: "gaussian_low", label: "高斯低通滤波" }
    ],
    mean: [
        { value: "arithmetic_mean", label: "算术均值" },
        { value: "geometric_mean", label: "几何均值" },
        { value: "harmonic_mean", label: "谐波均值" }
    ],
    sort: [
        { value: "max_filter", label: "最大值" },
        { value: "min_filter", label: "最小值" },
        { value: "median_filter", label: "中值" }
    ],
    select: [
        { value: "low_pass", label: "低通" },
        { value: "high_pass", label: "高通" },
        { value: "band_pass", label: "带通" },
        { value: "band_stop", label: "带阻" }
    ],
    noise: [
        { value: "gaussian", label: "高斯噪声" },
        { value: "salt_pepper", label: "椒盐噪声" }
    ],
    // 锐化
    sharpen_space: [
        { value: "roberts", label: "Roberts梯度算子" },
        { value: "sobel", label: "Sobel梯度算子" },
        { value: "prewitt", label: "Prewitt梯度算子" },
        { value: "laplacian", label: "Laplacian梯度算子" }
    ],
    sharpen_frequency: [
        { value: "ideal_high", label: "理想高通滤波" },
        { value: "butterworth_high", label: "巴特沃斯高通滤波" },
        { value: "gaussian_high", label: "高斯高通滤波" }
    ]
};

// 初始化
document.addEventListener("DOMContentLoaded", function () {
    const mainCategory = document.getElementById("main-category");
    const subCategory = document.getElementById("sub-category");
    const paramsArea = document.getElementById("params-area");
    const uploadArea = document.getElementById("upload-area");
    const form = document.getElementById("operation-form");
    const originalImg1 = document.getElementById("original-img1");
    const originalImg2 = document.getElementById("original-img2");
    const resultImg = document.getElementById("result-img");

    // 初始化二级分类
    function updateSubCategory() {
        const mainVal = mainCategory.value;
        subCategory.innerHTML = "";
        subCategoryMap[mainVal].forEach(opt => {
            const option = document.createElement("option");
            option.value = opt.value;
            option.textContent = opt.label;
            subCategory.appendChild(option);
        });
        updateParamsAndUpload();
    }

    // 根据功能显示参数和上传框
    function updateParamsAndUpload() {
        paramsArea.innerHTML = "";
        uploadArea.innerHTML = "";
        const mainVal = mainCategory.value;
        const subVal = subCategory.value;

        // 重置图片显示
        originalImg1.style.display = "none";
        originalImg2.style.display = "none";
        originalImg1.src = "";
        originalImg2.src = "";
        resultImg.src = "";

        // 上传框
        if (mainVal === "arithmetic" && ["add", "subtract", "multiply"].includes(subVal)) {
            uploadArea.innerHTML = `
                <label>上传图片1</label>
                <input type="file" name="file1" accept="image/*" required onchange="previewImage(this, 'original-img1')">
                <label>上传图片2</label>
                <input type="file" name="file2" accept="image/*" required onchange="previewImage(this, 'original-img2')">
            `;
            originalImg1.style.display = "block";
            originalImg2.style.display = "block";
        } else {
            uploadArea.innerHTML = `
                <label>上传图片</label>
                <input type="file" name="file" accept="image/*" required onchange="previewImage(this, 'original-img1')">
            `;
            originalImg1.style.display = "block";
        }

        // 参数区
        if (mainVal === "face" && subVal === "recognition") {
            paramsArea.innerHTML = `
                <div id="face-result" style="margin-top: 20px;">
                    <h3>识别结果</h3>
                    <div class="result-container" style="display: flex; justify-content: center;">
                        <div class="right-result" style="text-align: center;">
                            <h4>匹配结果</h4>
                            <p id="face-name" style="margin: 10px 0;"></p>
                            <img id="matched-face" src="" alt="匹配的人脸" style="max-width: 200px; display: none;" />
                        </div>
                    </div>
                </div>
            `;
        }
        // 图像几何变换
        if (mainVal === "arithmetic" && subVal === "geometric") {
            paramsArea.innerHTML = `
                <label>变换类型</label>
                <select name="transform_type" id="transform_type">
                    <option value="scale">缩放</option>
                    <option value="translate">平移</option>
                    <option value="rotate">旋转</option>
                    <option value="flip">翻转</option>
                </select>
                <div id="geometric-params"></div>
            `;
            document.getElementById("transform_type").addEventListener("change", updateGeometricParams);
            updateGeometricParams();
        }
        // 仿射变换
        if (mainVal === "arithmetic" && subVal === "affine") {
            paramsArea.innerHTML = `
                <label>源点(src_pts) [格式: [[x1,y1],[x2,y2],[x3,y3]]]</label>
                <input type="text" name="src_pts" value="[[0,0],[100,0],[0,100]]" required>
                <label>目标点(dst_pts) [格式: [[x1,y1],[x2,y2],[x3,y3]]]</label>
                <input type="text" name="dst_pts" value="[[10,10],[120,20],[20,120]]" required>
            `;
        }
        // 线性变换
        if (mainVal === "enhance" && subVal === "linear") {
            paramsArea.innerHTML = `
                <label>对比度(alpha)</label>
                <input type="number" name="alpha" value="1.0" step="0.1">
                <label>亮度(beta)</label>
                <input type="number" name="beta" value="0" step="1">
            `;
        }
        // 边缘检测
        if (mainVal === "segment" && subVal === "edge") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="method">
                    ${detailOptions.edge.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
            `;
        }
        // 线性变化检测
        if (mainVal === "segment" && subVal === "line") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="method" id="line-method">
                    ${detailOptions.line.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <label>阈值</label>
                <input type="number" name="threshold" value="150">
                <div id="line-extra"></div>
            `;
            document.getElementById("line-method").addEventListener("change", function () {
                const val = this.value;
                const extra = document.getElementById("line-extra");
                if (val === "houghp") {
                    extra.innerHTML = `
                        <label>最小线长</label>
                        <input type="number" name="min_line_length" value="50">
                        <label>最大线间隙</label>
                        <input type="number" name="max_line_gap" value="10">
                    `;
                } else {
                    extra.innerHTML = "";
                }
            });
        }
        // 空域平滑
        if (mainVal === "smooth" && subVal === "space") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation">
                    ${detailOptions.space.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
            `;
        }
        // 频域平滑
        if (mainVal === "smooth" && subVal === "frequency") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation" id="freq-method">
                    ${detailOptions.frequency.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <label>D0</label>
                <input type="number" name="D0" value="20">
                <label>n</label>
                <input type="number" name="n" value="2">
            `;
        }
        // 空域锐化
        if (mainVal === "sharpen" && subVal === "space") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation">
                    ${detailOptions.sharpen_space.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
            `;
        }
        // 频域锐化
        if (mainVal === "sharpen" && subVal === "frequency") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation" id="sharpen-freq-method">
                    ${detailOptions.sharpen_frequency.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <label>D0</label>
                <input type="number" name="D0" value="40">
                <label>n</label>
                <input type="number" name="n" value="2">
            `;
        }
        // 数学形态学
        if (mainVal === "morphology") {
            paramsArea.innerHTML = `
                <label>结构元类型</label>
                <select name="kernel_shape">
                    <option value="rect">矩形</option>
                    <option value="ellipse">椭圆</option>
                    <option value="cross">交叉</option>
                </select>
                <label>结构元大小</label>
                <input type="number" name="kernel_size" value="5">
                <label>迭代次数</label>
                <input type="number" name="iterations" value="1">
                <label>二值化阈值(可选)</label>
                <input type="number" name="threshold" value="">
            `;
        }
        // 噪声模拟
        if (mainVal === "restore" && subVal === "noise") {
            paramsArea.innerHTML = `
                <label>噪声类型</label>
                <select name="noise_type" id="noise-type">
                    ${detailOptions.noise.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <div id="noise-params"></div>
            `;
            document.getElementById("noise-type").addEventListener("change", updateNoiseParams);
            updateNoiseParams();
        }
        // 均值滤波
        if (mainVal === "restore" && subVal === "mean") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation">
                    ${detailOptions.mean.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <label>核大小</label>
                <input type="number" name="kernel_size" value="3">
            `;
        }
        // 排序滤波
        if (mainVal === "restore" && subVal === "sort") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation">
                    ${detailOptions.sort.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <label>核大小</label>
                <input type="number" name="kernel_size" value="3">
            `;
        }
        // 选择滤波
        if (mainVal === "restore" && subVal === "select") {
            paramsArea.innerHTML = `
                <label>方法</label>
                <select name="operation" id="select-method">
                    ${detailOptions.select.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join("")}
                </select>
                <div id="select-params"></div>
            `;
            document.getElementById("select-method").addEventListener("change", updateSelectParams);
            updateSelectParams();
        }
    }

    // 动态参数区：几何变换
    function updateGeometricParams() {
        const type = document.getElementById("transform_type").value;
        const area = document.getElementById("geometric-params");
        if (type === "scale") {
            area.innerHTML = `
                <label>fx(缩放x)</label>
                <input type="number" name="fx" value="1.0" step="0.1">
                <label>fy(缩放y)</label>
                <input type="number" name="fy" value="1.0" step="0.1">
            `;
        } else if (type === "translate") {
            area.innerHTML = `
                <label>tx(水平平移)</label>
                <input type="number" name="tx" value="0">
                <label>ty(垂直平移)</label>
                <input type="number" name="ty" value="0">
            `;
        } else if (type === "rotate") {
            area.innerHTML = `
                <label>角度(angle)</label>
                <input type="number" name="angle" value="0">
                <label>缩放(scale)</label>
                <input type="number" name="scale" value="1.0" step="0.1">
            `;
        } else if (type === "flip") {
            area.innerHTML = `
                <label>翻转代码(code)</label>
                <select name="code">
                    <option value="0">垂直</option>
                    <option value="1">水平</option>
                    <option value="-1">水平+垂直</option>
                </select>
            `;
        }
    }

    // 动态参数区：噪声模拟
    function updateNoiseParams() {
        const type = document.getElementById("noise-type").value;
        const area = document.getElementById("noise-params");
        if (type === "gaussian") {
            area.innerHTML = `
                <label>均值(mean)</label>
                <input type="number" name="mean" value="0" step="0.01">
                <label>方差(var)</label>
                <input type="number" name="var" value="0.1" step="0.01">
            `;
        } else if (type === "salt_pepper") {
            area.innerHTML = `
                <label>盐概率(salt_prob)</label>
                <input type="number" name="salt_prob" value="0.05" step="0.01">
                <label>椒概率(pepper_prob)</label>
                <input type="number" name="pepper_prob" value="0.05" step="0.01">
            `;
        }
    }

    // 动态参数区：选择滤波
    function updateSelectParams() {
        const type = document.getElementById("select-method").value;
        const area = document.getElementById("select-params");
        if (type === "low_pass" || type === "high_pass") {
            area.innerHTML = `
                <label>阈值</label>
                <input type="number" name="threshold" value="128">
            `;
        } else if (type === "band_pass" || type === "band_stop") {
            area.innerHTML = `
                <label>最小值</label>
                <input type="number" name="min_val" value="50">
                <label>最大值</label>
                <input type="number" name="max_val" value="200">
            `;
        }
    }

    // 预览图片
    window.previewImage = function(input, imgId) {
        const img = document.getElementById(imgId);
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
                img.style.display = "block";
            };
            reader.readAsDataURL(input.files[0]);
        }
    };

    // 监听分类变化
    mainCategory.addEventListener("change", updateSubCategory);
    subCategory.addEventListener("change", updateParamsAndUpload);

    // 初始化
    updateSubCategory();

    // 处理表单提交
    form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const mainVal = mainCategory.value;
        const subVal = subCategory.value;

        // 显示加载状态
        const submitBtn = document.getElementById("submit-btn");
        submitBtn.disabled = true;
        submitBtn.textContent = "处理中...";

        try {
            // 人脸识别处理
            if (mainVal === "face" && subVal === "recognition") {
                const formData = new FormData(form);
                const response = await fetch("/api/face/recognition", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();

                if (result.error) {
                    alert(result.error);
                    return;
                }

                // 显示原图
                const fileInput = form.querySelector('input[type="file"]');
                originalImg1.src = URL.createObjectURL(fileInput.files[0]);

                // 显示处理后的图像
                resultImg.src = result.result + "?t=" + new Date().getTime();

                // 更新识别结果显示
                const faceNameElement = document.getElementById("face-name");
                const matchedFaceElement = document.getElementById("matched-face");

                if (result.matched_name) {
                    // 找到匹配的人脸
                    faceNameElement.textContent = `识别结果：${result.matched_name}`;
                    if (result.matched_face) {
                        matchedFaceElement.src = result.matched_face + "?t=" + new Date().getTime();
                        matchedFaceElement.style.display = "block";
                    }
                } else {
                    // 检测到人脸但未匹配
                    faceNameElement.textContent = "识别结果：人脸未录入";
                    matchedFaceElement.style.display = "none";
                }

                return;
            }

            // 去雾处理
            if (mainVal === "defogging") {
                const formData = new FormData(form);
                const response = await fetch("/api/defogging", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();

                if (result.error) {
                    alert(result.error);
                    return;
                }

                // 显示原图和处理后的图片
                const fileInput = form.querySelector('input[type="file"]');
                originalImg1.src = URL.createObjectURL(fileInput.files[0]);
                resultImg.src = result.result + "?t=" + new Date().getTime();
                return;
            }

            // 其他图像处理...
            let url = "";
            let formData = new FormData(form);
            if (mainVal === "arithmetic") {
                if (["add", "subtract", "multiply"].includes(subVal)) {
                    url = "/api/arithmetic";
                    formData.append("op_type", subVal);
                } else if (subVal === "geometric") {
                    url = "/api/geometric";
                } else if (subVal === "affine") {
                    url = "/api/affine";
                } else if (subVal === "fourier") {
                    url = "/api/fourier";
                }
            } else if (mainVal === "enhance") {
                url = "/api/enhance";
                formData.append("method", subVal);
            } else if (mainVal === "segment") {
                if (subVal === "edge") {
                    url = "/api/edge";
                } else if (subVal === "line") {
                    url = "/api/line";
                }
            } else if (mainVal === "smooth") {
                if (subVal === "space") {
                    url = "/api/smooth/space";
                } else if (subVal === "frequency") {
                    url = "/api/smooth/frequency";
                }
            } else if (mainVal === "sharpen") {
                if (subVal === "space") {
                    url = "/api/sharpen/space";
                } else if (subVal === "frequency") {
                    url = "/api/sharpen/frequency";
                }
            } else if (mainVal === "morphology") {
                url = "/api/morphology";
                formData.append("operation", subVal);
            } else if (mainVal === "restore") {
                if (subVal === "noise") {
                    url = "/api/noise";
                } else if (subVal === "mean") {
                    url = "/api/filter/mean";
                } else if (subVal === "sort") {
                    url = "/api/filter/sort";
                } else if (subVal === "select") {
                    url = "/api/filter/select";
                }
            }

            const response = await fetch(url, {
                method: "POST",
                body: formData
            });
            const result = await response.json();

            if (result.error) {
                alert(result.error);
                return;
            }

            // 显示原图和处理后的图片
            if (result.result) {
                const file = form.querySelector('input[type="file"]');
                if (file) {
                    originalImg1.src = URL.createObjectURL(file.files[0]);
                }
                resultImg.src = `${result.result}?t=${new Date().getTime()}`;
            }

        } catch (error) {
            console.error("Error:", error);
            alert("处理失败，请重试");
        } finally {
            // 恢复提交按钮
            submitBtn.disabled = false;
            submitBtn.textContent = "提交";
        }
    });
});