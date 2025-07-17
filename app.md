# 林业害虫检测与识别系统

一个基于YOLO深度学习目标检测技术的林业害虫检测与识别系统，能够自动识别图像中的害虫种类并提供详细信息，支持多种害虫的实时检测与分类。

## 功能特点

- **多模型支持**：集成YOLOv5和YOLOv12两种先进的目标检测模型
- **多种害虫识别**：支持32种常见林业害虫的精准识别
- **详细信息展示**：提供害虫的中英文名称、分类信息和置信度
- **可视化结果**：自动在图像上标注检测结果，包括边界框和标签
- **RESTful API**：提供标准化的API接口，便于集成到其他系统
- **用户认证系统**：支持用户注册、登录功能
- **日志记录**：详细的系统日志，便于问题排查和性能优化
- **跨平台支持**：支持Windows和Linux系统部署

## 系统架构

```
├── app.py              # Flask应用主入口，包含主要路由和配置
├── auth.py             # 用户认证模块
├── detect.py           # YOLOv5检测实现，包含目标检测核心逻辑
├── v12_detection.py    # YOLOv12检测实现，提供更高级的检测功能
├── models/             # 预训练模型目录
│   └── experimental.py # 模型加载和处理工具
├── static/             # 静态资源
│   └── images/         # 检测结果图像存储目录
├── templates/          # 网页模板
├── uploads/            # 用户上传文件临时存储
├── utils/              # 工具函数
│   ├── dataloaders.py  # 数据加载工具
│   ├── general.py      # 通用工具函数
│   ├── plots.py        # 绘图工具
│   └── torch_utils.py  # PyTorch相关工具
├── ultralytics/        # YOLO相关代码
└── output/             # YOLOv12检测结果输出目录
```

## 技术栈

- **后端框架**：Flask
- **深度学习框架**：PyTorch, Ultralytics YOLO
- **图像处理**：OpenCV, PIL
- **数据库**：MySQL
- **用户认证**：Flask-Login
- **前端技术**：HTML, CSS, JavaScript
- **API文档**：RESTful API

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.7+
- CUDA 11.0+ (推荐用于GPU加速)
- Flask 2.0+
- MySQL 5.7+
- OpenCV 4.5+

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/your-username/yoloflask.git
cd yoloflask
```

2. 创建并激活虚拟环境（可选但推荐）：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载预训练模型：
```bash
# YOLOv5模型
mkdir -p resource/models
wget https://example.com/models/best.pt -O resource/models/best.pt

# YOLOv12模型
mkdir -p v12
wget https://example.com/models/yolov12.pt -O v12/yolov12.pt
```

5. 配置数据库：
```bash
# 创建MySQL数据库
mysql -u root -p
CREATE DATABASE databug;
USE databug;
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);
```

6. 修改数据库配置：
   在`app.py`中更新MySQL连接信息：
   ```python
   app.config['MYSQL_HOST'] = 'localhost'
   app.config['MYSQL_USER'] = 'your_username'
   app.config['MYSQL_PASSWORD'] = 'your_password'
   app.config['MYSQL_DB'] = 'databug'
   ```

7. 创建必要的目录：
```bash
mkdir -p uploads static/images output
```

8. 启动服务：
```bash
python app.py
```

服务将在 http://localhost:5000 启动。

## 使用说明

### Web界面

1. 访问 `http://localhost:5000`
2. 注册/登录账户
3. 上传包含害虫的图像
4. 选择检测模型（YOLOv5或YOLOv12）
5. 查看检测结果，包括害虫类别、置信度和标注图像

## API接口

#### 1. YOLOv5检测接口
```
POST /api/detect/0
Content-Type: multipart/form-data
参数: image (文件)

响应:
{
  "status": "success",
  "results": [
    {
      "label": "011.exc_Adult",
      "chinese_name": "二星蝽",
      "english_name": "Eysacoris guttiger",
      "confidence": 0.92,
      "bbox": [100, 200, 300, 400],
      "class_id": 0
    }
  ],
  "image_url": "/uploads/image.jpg",
  "result_image_url": "/static/images/result_1234567890_image.jpg",
  "processing_time": 0.45,
  "image_size": {"width": 640, "height": 480}
}
```

#### 2. YOLOv12检测接口
```
POST /api/v12/detect
Content-Type: multipart/form-data
参数: image (文件)

响应:
{
  "status": "success",
  "model": "yolov12",
  "detections": [
    {
      "bbox": [100, 200, 300, 400],
      "confidence": 0.95,
      "class": 0,
      "class_name": "011.exc_Adult"
    }
  ]
}
```

#### 3. 用户注册接口
```
POST /register
Content-Type: application/json
{
  "username": "user123",
  "phone": "13800138000",
  "password": "password123"
}

响应:
{
  "status": "success",
  "message": "注册成功"
}
```

#### 4. 用户登录接口
```
POST /login
Content-Type: application/json
{
  "username": "user123",
  "password": "password123"
}

响应:
{
  "status": "success",
  "data": {
    "user": {
      "id": 1,
      "username": "user123",
      "phone": "13800138000"
    }
  }
}
```

## 害虫类别列表

系统支持检测的害虫种类包括：

| ID | 标签 | 中文名 | 英文名 |
|----|------|--------|--------|
| 0 | 011.exc_Adult | 二星蝽 | Eysacoris guttiger |
| 1 | 012.ybtn_Adult | 云斑天牛 | Batocera horsfieldi |
| 2 | 013.gjxtn_Adult | 光肩星天牛 | Anoplophora glabripennis |
| 3 | 014.bdgclc_Adult | 八点广翅蜡蝉 | Ricaniaspeculum Walker |
| 5 | 032.mtn_Adult | 墨天牛 | Monochamus alternatus |
| 6 | 045.xlyc_Adult | 小绿叶蝉 | Empoasca flavescens |
| 7 | 053.bce_larva | 扁刺蛾 | Thosea sinensis |
| 8 | 054.sze_Adult | 扇舟蛾 | Clostera anachoreta |
| 9 | 058.bylc_Adult | 斑衣蜡蝉 | Lycorma delicatula |
| 10 | 060.xmye_Adult | 旋目夜蛾 | Speiredonia retorta |
| 11 | 065.llyj_Adult | 柳蓝叶甲 | plagiodera versicolora |
| 12 | 067.tzm_Adult | 桃蛀螟 | Conogethes punctiferalis |
| 13 | 069.stn_Adult | 桑天牛 | Apriona germari |
| 15 | 009.ewze_Adult | 二尾舟蛾 | Cerura menciana |
| 16 | 010.ewze_larva | 二尾舟蛾(幼虫) | Cerura menciana(larva) |
| 17 | 075.ydfd_Adult | 玉带凤蝶 | Papilio polytes |
| 18 | 080.bxhjg_Adult | 白星花金龟 | Brevitarsis |
| 19 | 087.belc_Adult | 碧蛾蜡蝉 | Geisha distinctissima |
| 20 | 089.djyc_Adult | 稻棘缘蝽 | Cletus punctiger Dallas |
| 21 | 096.hyde_Adult | 红缘灯蛾 | Amsacta lactinea |
| 22 | 099.hjtn_Adult | 红颈天牛 | Aromia bungii |
| 23 | 103.lce_Adult | 绿刺蛾 | Parasa tessellata |
| 24 | 104.lce_larva | 绿刺蛾(幼虫) | Parasa tessellata(larva) |
| 25 | 105.mgbe_Adult | 美国白蛾 | Hyphantria cunea |
| 26 | 109.ccc_Adult | 茶翅蝽 | Halyomorpha halys |
| 28 | 111.cfd_Adult | 菜粉蝶 | Pieris rapae |

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证

MIT License

Copyright (c) 2023 Your Name

Permission is hereby granted...
