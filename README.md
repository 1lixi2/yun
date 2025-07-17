# 林业害虫识别系统

## 项目简介

林业害虫识别系统是一款专业的移动应用，旨在帮助林业工作者、研究人员和爱好者快速识别各类林业害虫，并提供相关的防治建议和知识支持。该系统结合了先进的图像识别技术和人工智能，提供了在线识别和离线识别两种模式，满足不同场景下的使用需求。

## 功能特点

### 1. 害虫识别
- **在线识别**：通过API上传图片进行云端识别，提供高精度的识别结果
- **离线识别**：使用TensorFlow.js在本地进行识别，无需网络连接，适合野外工作环境
- **识别详情**：展示害虫名称、英文名称、置信度、位置等详细信息

### 2. 智能咨询
- **AI聊天助手**：内置专业的林业害虫知识助手，可回答用户关于林业害虫的各种问题
- **防治建议**：基于识别结果，通过DeepSeek API提供专业的防治方法和建议

### 3. 社交分享
- **结果分享**：支持将识别结果分享到社交媒体
- **链接分享**：生成分享链接和二维码，方便与他人共享识别结果

### 4. 用户系统
- **用户注册/登录**：支持用户账户管理
- **个人资料**：用户可以管理个人信息和设置

## 技术栈

- **前端框架**：Vue.js
- **UI组件**：uni-app组件
- **图像识别**：TensorFlow.js (离线识别)
- **AI对话**：DeepSeek API
- **状态管理**：Vuex

## 安装和使用

### 环境要求
- Node.js 12.0+
- npm 或 yarn

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/1lixi2/yun.git
```

2. 安装依赖
```bash
npm install
# 或
yarn install
```

3. 运行开发服务器
```bash
npm run dev
# 或
yarn dev
```

4. 构建生产版本
```bash
npm run build
# 或
yarn build
```

## 项目结构

```
forestry-pest-identification/
├── components/          # 组件目录
│   └── AiChat/          # AI聊天组件
├── config/              # 配置文件
│   └── api.js           # API配置
├── pages/               # 页面目录
│   ├── chat/            # 聊天页面
│   ├── gallery/         # 图库页面
│   ├── identify/        # 识别相关页面
│   ├── index/           # 首页
│   ├── lib/             # 库文件
│   ├── prevention/      # 防治页面
│   ├── profile/         # 用户资料页面
│   ├── register/        # 注册页面
│   ├── share/           # 分享页面
│   └── welcome/         # 欢迎页面
├── static/              # 静态资源
├── store/               # Vuex状态管理
│   └── user.js          # 用户状态管理
├── App.vue              # 应用主组件
├── main.js              # 应用入口文件
└── pages.json           # 页面配置
```

## 使用指南

### 在线识别
1. 点击首页的"在线识别"按钮
2. 选择或拍摄害虫图片
3. 上传图片后系统将自动进行识别
4. 查看识别结果和防治建议

### 离线识别
1. 点击首页的"离线识别"按钮
2. 选择或拍摄害虫图片
3. 系统将在本地进行识别处理
4. 查看识别结果和防治建议

### AI聊天
1. 点击悬浮的机器人图标进入聊天页面
2. 输入关于林业害虫的问题
3. AI助手将提供专业的回答和建议

## API配置

系统使用DeepSeek API提供AI对话功能，配置如下：

```javascript
export const deepseekConfig = {
  apiUrl: 'https://api.deepseek.com/v1/chat/completions',
  apiKey: 'your-api-key',  // 请替换为您的API密钥
  model: 'deepseek-chat',
  temperature: 0.7,
  maxTokens: 800,
  systemPrompt: '你是一个专业的林业害虫知识助手，可以回答用户关于林业害虫的各种问题，包括害虫识别、防治方法、生态影响等。请提供准确、专业的回答。'
};
```

## 贡献指南

我们欢迎并感谢任何形式的贡献！如果您想为项目做出贡献，请遵循以下步骤：

1. Fork 项目仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者 - [您的名字](mailto:your.email@example.com)

项目链接: [https://github.com/your-username/forestry-pest-identification](https://github.com/your-username/forestry-pest-identification)
