# 构建阶段
FROM python:3.9-slim-buster as builder

# 设置工作目录
WORKDIR /app

# 将 requirements.txt 复制到工作目录
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 运行时阶段
FROM python:3.9-alpine

# 设置工作目录
WORKDIR /app

# 从构建阶段复制安装的依赖和可执行文件
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# 将所有文件复制到工作目录
COPY . .

# 暴露应用程序运行的端口
EXPOSE 5000

# 定义环境变量 (这些在docker-compose.yml中已移除，但保留在Dockerfile中以防直接构建)
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# 启动 Flask 应用程序
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--timeout", "300", "main:app"]