# PyTorch 2.4.1 と互換性のあるCUDAバージョンを持つ公式イメージを指定
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# イメージ内に作成する作業ディレクトリを設定
WORKDIR /app

ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y  --no-install-recommends libgl1-mesa-glx libglib2.0-0 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 依存関係のインストール ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- ソースコードのコピー ---
# 実行に必要なソースコードと設定ファイルのみをコピーします。
COPY ./src ./src
COPY main.py .
COPY combine_results.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

# --- デフォルトの実行コマンド ---
# このイメージを使ってコンテナを起動した際に、デフォルトで実行されるコマンドです。
# KubernetesのJob定義で上書きされることが多いですが、適切なデフォルトを設定しておくと便利です。
#CMD ["python", "main.py"]
