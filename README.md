# CCR — Codex Responses → OpenAI Chat Completions Proxy

CCR 是一个 Rust 编写的反向代理，将 Codex 的 **Responses API** 请求转换为 OpenAI **Chat Completions API** 格式，转发到上游兼容 API，再将响应转回 Responses 格式返回给 Codex。

适合在 Codex 中使用非 OpenAI 官方的模型（如 DeepSeek、本地部署模型等）。

## 工作流程

```
Codex (Responses API)  -->  CCR (本地 :8180)  -->  上游 Chat Completions API
                         POST /v1/responses        (OpenAI / DeepSeek / 自定义)
```

1. Codex 以 wire_api = "responses" 模式发送请求到 CCR
2. CCR 将请求转换为 Chat Completions 格式并转发到上游
3. 上游返回 Chat Completions 格式的响应
4. CCR 将响应转换回 Responses 格式返回给 Codex

## 功能

- **请求转换**: 支持 string/array input、instructions、tool calls、reasoning、image 等多种输入格式
- **响应转换**: 将 Chat Completions 的 choices 转回 Responses 的 output 结构
- **流式支持**: 完整 SSE 流转换，含 keep-alive 心跳、超时控制、预检测与兜底机制
- **模型映射**: Codex 请求的 model 自动映射到上游模型名
- **Usage 注入**: 上游未返回 token 统计时，基于输出长度本地估算并注入
- **预检测**: 在发送流式响应前检测认证错误、空响应、配额不足等问题
- **结构化日志**: json/文本格式，按日期滚动保留
- **并行工具调用**: 支持 Codex 的多 tool call 合并与拆分

## 快速开始

### 编译

```bash
cargo build --release
```

二进制文件位于 	target/release/ccr（Windows 上为 ccr.exe）。

### 配置

从示例文件创建配置：

```bash
copy config.example.toml config.toml
```

编辑 config.toml，填入上游 API 地址与密钥：

```	toml
[server]
host = "0.0.0.0"
port = 8180

[upstream]
url = "https://api.deepseek.com/v1/chat/completions"
api_key = "sk-your-deepseek-key"

[upstream.model_mapping]
"gpt-5.4" = "deepseek-v4-pro"
"gpt-5.5" = "deepseek-v4-pro"

[streaming]
keepalive_interval_secs = 15
enable_usage_injection = true
```

### 运行

```bash
./target/release/ccr config.toml
```

验证服务是否正常：

```bash
curl http://127.0.0.1:8180/health
# 返回: OK
```

## Codex 配置

要让 Codex 使用 CCR，需要修改 Codex 的配置文件。

### 找到 Codex 配置

Codex 配置文件路径取决于平台：

- **Windows**: %APPDATA%\codex\config.toml
- **macOS**: ~/Library/Application Support/com.openai.codex/config.toml

### 添加自定义模型提供商

在 Codex 配置文件中添加或修改如下内容：

```	toml
# -- 关键配置：指定模型提供商和模型 --
model_provider = "custom"
model = "gpt-5.4"
model_reasoning_effort = "high"
disable_response_storage = true

# -- 定义自定义 provider --
[model_providers.custom]
name = "custom"
wire_api = "responses"
requires_openai_auth = true
base_url = "http://127.0.0.1:8180/v1"
```

配置要点：

- model_provider 设为自定义 provider 的名字（如 "custom"）
- wire_api = "responses" 让 Codex 使用 Responses API 格式通信
- requires_openai_auth = true Codex 会发送 OpenAI 认证头（CCR 忽略它，使用自己的上游 API key）
- base_url 指向 CCR 的地址，Codex 会自动拼接为 http://127.0.0.1:8180/v1/responses
- model 填 CCR 配置中 model_mapping 映射的源模型名（如 "gpt-5.4"）

### 信任项目目录（可选）

如果项目在本地且需要使用工具调用（如文件读写、shell 命令），可以添加项目信任：

```	toml
[projects.'{your_path}/project/my-project']
trust_level = "trusted"
```

注意：Codex 配置中的路径分隔符使用正斜杠 /，即使 Windows 也用正斜杠（如 {your_path}/project/my-project）。

### 参考示例

完整示例见example/codex/config.toml。

## 配置参考

### server

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| host | string | "0.0.0.0" | 监听地址 |
| port | 数字 | 8180 | 监听端口 |
| connect_timeout_secs | 数字 | 30 | 连接上游超时（秒） |
| request_timeout_secs | 数字 | 1800 | 单个请求总超时（秒） |
| max_body_size | 数字 | 10485760 | 最大请求体大小（字节） |

### upstream

| 字段 | 类型 | 说明 |
|------|------|------|
| url | string | 上游 Chat Completions API 地址 |
| api_key | string | 上游 API 密钥 |
| model_mapping | table | 模型名映射（Codex model → 上游 model） |
| extra_headers | table | 附加的自定义请求头 |

### streaming

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| keepalive_interval_secs | 数字 | 15 | SSE keep-alive 心跳间隔 |
| enable_usage_injection | bool | true | 未返回 usage 时注入本地估算 |
| preflight_timeout_secs | 数字 | 120 | 流预检测超时 |
| total_timeout_secs | 数字 | 3600 | 流式响应的最大时长 |
| enable_preflight | bool | true | 启用流预检测（检测认证/配额/空响应错误） |

### logging

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| level | string | "info" | 日志级别（trace/debug/info/warn/error） |
| json | bool | false | 是否输出 JSON 格式日志 |
| dir | string | "logs" | 日志文件目录（留空不写文件） |
| file_prefix | string | "ccr" | 日志文件名前缀 |
| rotation | bool | true | 是否按日期滚动 |
| retention_days | 数字 | 3 | 日志保留天数 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /v1/responses | 处理 Codex Responses 请求 |
| GET | /health | 健康检查 |

## 请求转换详情

CCR 处理以下 Codex Responses 输入类型：

- input_text: 拼接为纯文本字符串
- input_image: 转为 Chat Completions 的 image_url 格式
- function_call / function_call_output: 转为 tool_calls / tool 消息
- reasoning: 提取 summary_text 转为 reasoning_content
- instructions: 作为 system 消息插入
- 	tools: 透传并规范化参数格式

所有支持详情参见 	ests/request_converter.rs 中的测试用例。

## 故障排查

**Codex 返回错误**: 查看 CCR 日志（logs/ 目录），常见问题包括上游 API key 无效、模型映射缺失、网络不通。

**流式响应中断**: 检查 streaming 配置中的超时设置，确认上游支持 SSE 流式输出。

**模型不响应工具调用**: 确认上游模型支持 function calling / tool use，并在 Codex 配置中保留了工具定义。wire_api = "responses" 模式下 Codex 会自动转换工具格式。
