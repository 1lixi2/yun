from flask import Flask, request, jsonify, Response, send_from_directory, redirect, url_for, render_template, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pymysql
import torch
import sys
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import time
import logging
import traceback
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log",encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("yolo-flask")

i = 0

# 修复PosixPath问题的函数
def fix_posix_path_issue():
    import pathlib
    import platform
    
    if platform.system() == 'Windows':
        logger.info("检测到Windows系统，应用PosixPath修复")
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        logger.info(f"当前系统: {platform.system()}, 无需PosixPath修复")


try:
    fix_posix_path_issue()
    logger.info("PosixPath修复应用成功")
except Exception as e:
    logger.error(f"PosixPath修复失败: {str(e)}")
    logger.error(traceback.format_exc())

# 导入YOLO相关模块
from models.experimental import attempt_load
from utils.general import non_max_suppression

app = Flask(__name__)
CORS(app)  # 启用CORS

# 请求前中间件
@app.before_request
def before_request():
    request.start_time = time.time()
    logger.info(f"收到请求: {request.method} {request.path}")
    logger.debug(f"请求头: {dict(request.headers)}")
    
    # 记录请求体信息
    if request.content_type and 'application/json' in request.content_type:
        try:
            data = request.get_json(silent=True)
            if data:
                # 不记录图像数据以避免日志过大
                if 'image' in data:
                    data['image'] = f"[BASE64_IMAGE: {len(data['image'])} bytes]"
                logger.debug(f"请求体: {data}")
        except Exception as e:
            logger.warning(f"无法解析JSON请求体: {str(e)}")

# 请求后中间件
@app.after_request
def after_request(response):
    # 计算请求处理时间
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"请求处理完成: {request.method} {request.path} - 状态码: {response.status_code} - 耗时: {duration:.4f}秒")
    
    return response

# 错误处理器
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未捕获的异常: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "status": "error",
        "message": str(e),
        "error_type": type(e).__name__
    }), 500

# 配置
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 加载模型
start_time = time.time()
logger.info("开始初始化模型...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# 检查模型文件路径
model_paths = [
    Path('best.pt'),                    # 根目录
    Path('resource/models/best.pt'),    # 原始路径

]

model_path = None
for path in model_paths:
    if path.exists():
        model_path = path
        logger.info(f"找到模型文件: {path.absolute()}")
        break
    else:
        logger.debug(f"模型文件不存在: {path.absolute()}")

if model_path is None:
    logger.error("错误: 未找到模型文件，请确保模型文件存在于以下路径之一:")
    for path in model_paths:
        logger.error(f"  - {path.absolute()}")
    sys.exit(1)

try:
    logger.info(f"正在加载模型: {model_path}")
    model = attempt_load(str(model_path), device=device)
    load_time = time.time() - start_time
    logger.info(f"模型加载成功! 耗时: {load_time:.2f}秒")
    
    # 记录模型信息
    names = model.module.names if hasattr(model, 'module') else model.names
    logger.info(f"模型类别: {names}")
    logger.info(f"模型结构: {type(model).__name__}")
    
    # 记录系统信息
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# 害虫类别信息
train_set_class_name = [
        [0, "011.exc_Adult", "二星蝽","Eysacoris guttiger"],
        [1, "012.ybtn_Adult", "云斑天牛", "Batocera horsfieldi"],
        [2, "013.gjxtn_Adult", "光肩星天牛", "Anoplophora glabripennis"],
        [3, "014.bdgclc_Adult", "八点广翅蜡蝉", "Ricaniaspeculum Walker"],
        [4, "028.sxdde_Adult", "", ""],
        [5, "032.mtn_Adult", "墨天牛", "Monochamus alternatus"],
        [6, "045.xlyc_Adult", "小绿叶蝉", "Empoasca flavescens"],
        [7, "053.bce_larva", "扁刺蛾", "Thosea sinensis"],
        [8, "054.sze_Adult", "扇舟蛾", "Clostera anachoreta"],
        [9, "058.bylc_Adult", "斑衣蜡蝉", "Lycorma delicatula"],
        [10, "060.xmye_Adult", "旋目夜蛾", "Speiredonia retorta"],
        [11, "065.llyj_Adult", "柳蓝叶甲", "plagiodera versicolora"],
        [12, "067.tzm_Adult", "桃蛀螟", "Conogethes punctiferalis"],
        [13, "069.stn_Adult", "桑天牛", "Apriona germari"],
        [14, "071.hwcc_Adult", "", ""],
        [15, "009.ewze_Adult", "二尾舟蛾", "Cerura menciana"],
        [16, "010.ewze_larva", "二尾舟蛾(幼虫)", "Cerura menciana(larva)"],
        [17, "075.ydfd_Adult", "玉带凤蝶", "Papilio polytes"],
        [18, "080.bxhjg_Adult", "白星花金龟", "Brevitarsis"],
        [19, "087.belc_Adult", "碧蛾蜡蝉", "Geisha distinctissima"],
        [20, "089.djyc_Adult", "稻棘缘蝽", "Cletus punctiger Dallas"],
        [21, "096.hyde_Adult", "红缘灯蛾", "Amsacta lactinea"],
        [22, "099.hjtn_Adult", "红颈天牛", "Aromia bungii"],
        [23, "103.lce_Adult", "绿刺蛾", "Parasa tessellata"],
        [24, "104.lce_larva", "绿刺蛾", "Parasa tessellata(larva)"],
        [25, "105.mgbe_Adult", "美国白蛾", "Hyphantria cunea"],
        [26, "109.ccc_Adult", "茶翅蝽", "Halyomorpha halys"],
        [27, "110.clj_Adult", "", ""],
        [28, "111.cfd_Adult", "菜粉蝶", "Pieris rapae"],
        [29, "116.lg_Adult", "蝼蛄", "Gryllotalpa spps"],
        [30, "123.ctc_Adult", "赤条蝽", "Graphosoma rubrolineata"],
        [31, "132.mpc_Adult", "麻皮蝽", "Erthesina fullo"],
        [32, "140.hzc_Adult", "黑蚱蝉", "Cryptotympana atrata Fabricius"],
    ]

@app.route('/api/detect/<int:id>', methods=['POST'])
def detect_pest(id):
    request_id = getattr(request, 'request_id', f"req_{int(time.time() * 1000)}")
    logger.info(f"[{request_id}] 开始处理害虫检测请求")

    # 获取 id 参数
    if id is None:
        return jsonify({
            "status": "error",
            "message": "Missing id parameter"
        }), 400

    # 检查文件上传
    if 'image' not in request.files:
        logger.error(f"[{request_id}] 请求中没有image字段")
        logger.info(f"[{request_id}] 可用文件字段: {list(request.files.keys())}")
        return jsonify({
            "status": "error",
            "message": "No image uploaded",
            "expected_field": "image",
            "received_fields": list(request.files.keys())
        }), 400

    file = request.files['image']
    
    # 验证文件名
    if file.filename == '':
        logger.error(f"[{request_id}] 上传的文件名为空")
        return jsonify({
            "status": "error",
            "message": "No selected file",
            "received_file": bool(file)
        }), 400

    # 验证文件类型
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        logger.error(f"[{request_id}] 不支持的文件类型: {file.filename}")
        return jsonify({
            "status": "error",
            "message": "Unsupported file type",
            "allowed_types": list(allowed_extensions),
            "received_type": file.filename.split('.')[-1].lower() if '.' in file.filename else None
        }), 400

    try:
        # 保存上传文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"[{request_id}] 保存文件到: {filepath}")
        file.save(filepath)

        # 读取并预处理图像
        logger.info(f"[{request_id}] 开始处理图像")
        img = cv2.imread(filepath)
        if img is None:
            logger.error(f"[{request_id}] 无法读取图像文件: {filepath}")
            return jsonify({
                "status": "error",
                "message": "Cannot read image file"
            }), 400

        # 记录原始尺寸
        original_h, original_w = img.shape[:2]
        logger.info(f"[{request_id}] 原始图像尺寸: {original_w}x{original_h}")

        # 调整尺寸为64的倍数
        def make_divisible(x, divisor=64):
            return int(np.ceil(x / divisor)) * divisor

        new_w = make_divisible(original_w)
        new_h = make_divisible(original_h)
        
        if new_w != original_w or new_h != original_h:
            logger.info(f"[{request_id}] 调整图像尺寸到: {new_w}x{new_h}")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).to(device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        if id == 12:
            global i
            i += 1
            modelv12 = YOLO('../yoloflask/resource/models/bestv12.pt')
            logger.info(f"[{request_id}] 开始目标检测")
            detection_start = time.time()
            results = modelv12.predict(filepath, save=True, project="output", name=f"demo{i}")
            detection_time = time.time() - detection_start

            # 解析第一张图像的结果
            result = results[0]

            # 获取结构化数据
            for box in result.boxes:
                class_index = int(box.cls)
                class_name = train_set_class_name[class_index]
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()  # 获取边界框坐标

                print(f"检测到: {class_name} ({confidence:.2f})")
                print(f"位置: {bbox}")

            result = {
                "label": class_name[1],
                "chinese_name": class_name[2],
                "english_name": class_name[3],
                "confidence": confidence,
                "bbox": bbox,
                "class_id": class_name[0]
            }

            return jsonify({
                "status": "success",
                "results": result,
                "image_url": f"/uploads/{filename}",
                "result_image_url": f"/output/demo{i}/{filename}",
                "processing_time": detection_time,
                "image_size": {"width": img.shape[1],
                               "height": img.shape[0]}
            })

        # 执行检测
        logger.info(f"[{request_id}] 开始目标检测")
        detection_start = time.time()
        with torch.no_grad():
            pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        detection_time = time.time() - detection_start

        # 处理检测结果
        results = []
        # 创建图像副本用于绘制
        img_result = img.copy()

        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            class_id = int(cls)

            # 安全获取类别信息，处理未知类别ID
            try:
                # 在列表中查找匹配的类别ID
                found = False
                for item in train_set_class_name:
                    if item[0] == class_id:
                        label = item[1]
                        chinese_name = item[2]
                        english_name = item[3]
                        found = True
                        break

                if not found:
                    logger.warning(f"[{request_id}] 未知类别ID: {class_id}, 不在train_set_class_name列表中")
                    label = f"unknown_{class_id}"
                    chinese_name = f"未知类别_{class_id}"
                    english_name = f"Unknown_{class_id}"
            except Exception as e:
                logger.error(f"[{request_id}] 处理类别ID {class_id} 时出错: {str(e)}")
                label = f"error_{class_id}"
                chinese_name = f"错误类别_{class_id}"
                english_name = f"Error_{class_id}"

            result = {
                "label": label,
                "chinese_name": chinese_name,
                "english_name": english_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id
            }
            results.append(result)

            # 绘制边界框和标签
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 255, 0)  # 绿色边框
            thickness = 2
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, thickness)

            # 使用PIL绘制中文文本
            from PIL import Image, ImageDraw, ImageFont


            # 转换OpenCV图像为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)


            font_size = 20
            font = None

            # 常见中文字体列表
            font_list = [
                "simhei.ttf",                # 黑体
                "simsun.ttc",                # 宋体
                "msyh.ttc",                  # 微软雅黑
                "simkai.ttf",                # 楷体
                "C:/Windows/Fonts/simhei.ttf",  # 完整路径黑体
                "C:/Windows/Fonts/simsun.ttc",  # 完整路径宋体
                "C:/Windows/Fonts/msyh.ttc",    # 完整路径微软雅黑
                "C:/Windows/Fonts/simkai.ttf",  # 完整路径楷体
                "arial.ttf",                 # Arial 
                "C:/Windows/Fonts/arial.ttf"   # 完整路径Arial
            ]

            # 尝试加载字体
            for font_name in font_list:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    logger.info(f"成功加载字体: {font_name}")
                    break
                except IOError:
                    continue


            if font is None:
                logger.warning(f"无法加载任何中文字体，使用默认字体")
                font = ImageFont.load_default()

            # 添加标签文本
            text = f"{chinese_name} {conf:.2f}"

            # 使用font.getsize代替draw.textsize (兼容新版PIL)
            try:
                # 尝试使用getsize (旧版PIL)
                text_width, text_height = font.getsize(text)
            except AttributeError:
                try:
                    # 尝试使用getbbox (新版PIL)
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:

                    text_width = len(text) * font_size * 0.6
                    text_height = font_size + 4

            # 绘制文本背景
            draw.rectangle([(x1, y1 - text_height - 4), (x1 + text_width, y1)], fill=(0, 255, 0))

            # 绘制文本
            draw.text((x1, y1 - text_height - 2), text, fill=(255, 0, 0), font=font)

            # 转换回OpenCV图像
            img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        logger.info(f"[{request_id}] 检测完成, 耗时: {detection_time:.2f}秒, 检测到 {len(results)} 个目标")

        # 保存处理后的图片
        result_filename = f"result_{int(time.time())}_{os.path.basename(filename)}"
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        logger.info(f"[{request_id}] 保存处理后图片到: {result_filepath}")
        cv2.imwrite(result_filepath, img_result)

        return jsonify({
            "status": "success",
            "results": results,
            "image_url": f"/uploads/{filename}",
            "result_image_url": f"/static/images/{result_filename}",
            "processing_time": detection_time,
            "image_size": {"width": img.shape[1], "height": img.shape[0]}
        })

    except Exception as e:
        logger.error(f"[{request_id}] 检测过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }), 500


# 数据库配置
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Y520520.yun'
app.config['MYSQL_DB'] = 'databug'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 用于会话安全

# 微信登录配置
app.config['WECHAT_APP_ID'] = 'XXXXX'
app.config['WECHAT_APP_SECRET'] = 'XXXXX'
app.config['WECHAT_REDIRECT_URI'] = 'XXXXXXX'

# 初始化Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, phone):
        self.id = id
        self.username = username
        self.phone = phone

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            db=app.config['MYSQL_DB']
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, phone FROM users WHERE id = %s", (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(id=user_data[0], username=user_data[1], phone=user_data[2])
        conn.close()
        return None
    except Exception as e:
        logger.error(f"加载用户失败: {str(e)}")
        return None

# 配置静态文件目录
app.config['UPLOAD_FOLDER'] = 'D:/1/YOLO/yoloflask/static'

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/demo<int:id>/<path:filename>')
def static_files_v12(id, filename):
    directory = f"D:/1/YOLO/yoloflask/output/demo{id}"
    file_path = os.path.join(directory, filename)

    logger.info(f"请求静态文件: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"文件未找到: {file_path}")
        return jsonify({
            "status": "error",
            "message": f"文件未找到: {filename}",
            "directory": directory
        }), 404

    return send_from_directory(directory, filename)

@app.route('/register', methods=['POST'])
def register():
    # 验证请求内容类型
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "只支持JSON格式请求"
        }), 400

    # 解析请求数据
    data = request.get_json()
    username = data.get('username')
    phone = data.get('phone')
    password = data.get('password')

    # 验证必需字段
    if not all([username, phone, password]):
        return jsonify({
            "status": "error",
            "message": "用户名、手机号和密码不能为空"
        }), 400

    try:
        conn = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            db=app.config['MYSQL_DB']
        )
        with conn.cursor() as cursor:
            # 检查用户名和手机号是否已存在
            cursor.execute(
                "SELECT id FROM users WHERE username = %s OR phone = %s", 
                (username, phone)
            )
            if cursor.fetchone():
                return jsonify({
                    "status": "error",
                    "message": "用户名或手机号已存在"
                }), 400
            
            # 插入新用户
            cursor.execute(
                "INSERT INTO users (username, phone, password) VALUES (%s, %s, %s)",
                (username, phone, password)
            )
            conn.commit()
            
            return jsonify({
                "status": "success",
                "message": "注册成功"
            }), 200

    except pymysql.Error as e:
        logger.error(f"数据库错误: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "数据库错误"
        }), 500

    except Exception as e:
        logger.error(f"注册失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "注册失败"
        }), 500

    finally:
        if 'conn' in locals() and conn:
            conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    # 验证请求内容类型
    if not request.is_json:
        logger.error("非JSON格式请求")
        return jsonify({
            "status": "error",
            "code": "INVALID_CONTENT_TYPE",
            "message": "只支持application/json格式请求"
        }), 400

    # 解析请求数据
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # 验证必需字段
    if not username or not password:
        logger.error("缺少用户名或密码字段")
        return jsonify({
            "status": "error",
            "code": "MISSING_FIELDS",
            "message": "username和password字段不能为空"
        }), 400

    logger.info(f"开始处理用户登录: {username}")

    # 数据库查询
    try:
        conn = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            db=app.config['MYSQL_DB']
        )
        with conn.cursor() as cursor:
            # 查询用户
            cursor.execute(
                "SELECT id, username, phone, password FROM users WHERE username = %s",
                (username,)
            )
            user_data = cursor.fetchone()

            if not user_data:
                logger.warning(f"用户不存在: {username}")
                return jsonify({
                    "status": "error",
                    "code": "INVALID_CREDENTIALS",
                    "message": "用户名或密码错误"
                }), 401

            # 密码验证 
            if password != user_data[3]:
                logger.warning(f"密码不匹配: {username}")
                return jsonify({
                    "status": "error",
                    "code": "INVALID_CREDENTIALS",
                    "message": "用户名或密码错误"
                }), 401

            # 登录成功
            user = User(id=user_data[0], username=user_data[1], phone=user_data[2])
            login_user(user)
            
            logger.info(f"用户登录成功: {username}")
            return jsonify({
                "status": "success",
                "data": {
                    "user": {
                        "id": user_data[0],
                        "username": user_data[1],
                        "phone": user_data[2]
                    }
                }
            }), 200

    except pymysql.Error as db_error:
        logger.error(f"数据库错误: {str(db_error)}")
        return jsonify({
            "status": "error",
            "code": "DATABASE_ERROR",
            "message": "系统错误，请稍后再试"
        }), 500

    except Exception as e:
        logger.error(f"处理登录请求时发生异常: {str(e)}")
        return jsonify({
            "status": "error",
            "code": "SERVER_ERROR",
            "message": "服务器内部错误"
        }), 500

    finally:
        if 'conn' in locals() and conn:
            conn.close()

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已登出')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
