# 构建阶段
FROM docker.m.daocloud.io/library/python:3.11-slim AS builder
# 设置工作目录
WORKDIR /app

# 安装编译 C 扩展所需的构建工具（如 ujson / uvloop）
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

# 将 requirements.txt 复制到工作目录
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 运行时阶段（与构建阶段保持一致的 glibc 基础镜像，确保 C 扩展兼容）
FROM docker.m.daocloud.io/library/python:3.11-slim

# 设置工作目录
WORKDIR /app

# 从构建阶段复制安装的依赖和可执行文件
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 将所有文件复制到工作目录
COPY . .

# 暴露应用程序运行的端口
EXPOSE 5000

# 定义环境变量 (可通过 docker-compose.yml / .env 覆盖)
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=5000

# 启动 FastAPI 应用程序
CMD ["python", "app.py"]
