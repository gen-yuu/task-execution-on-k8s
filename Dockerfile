# 1. ベースイメージの選定
# NVIDIA公式のCUDA + cuDNN入りUbuntu22.04イメージを使用。これが最も確実。
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# 環境変数を設定（Pythonのバッファリング無効化など）
ENV PYTHONUNBUFFERED 1

# 2. 必要なツールとPythonをインストール
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python3.11へのシンボリックリンクを作成
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# 3. requirements.txtをコピーしてライブラリをインストール
WORKDIR /app
COPY requirements.txt .

# pipをアップグレードし、requirements.txtの内容をインストール
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 4. ソースコードをコピー
COPY . .

# このDockerfileを実行したときにデフォルトで実行されるコマンド (オプション)
# CMD ["python3", "run_single_task.py"]