# 使用 Python 3.13 作为基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml ./
COPY src/ ./src/
COPY main.py ./

# 安装 pip 和项目依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["python", "main.py"]

