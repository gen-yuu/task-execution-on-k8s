#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=task-runner-test
COMPOSE_FILE="./docker-compose.yml" 
FORCE_FLAG=false
BUILD_FLAG=false

# 引数の処理
while [[ "${1:-}" =~ ^- ]]; do
  case "$1" in
    -b|--build) 
      BUILD_FLAG=true
      shift ;;
    -f|--force)
      FORCE_FLAG=true
      shift ;;
    -h|--help)
      echo "Usage: $(basename "$0") [-f|--force]"
      echo "  (no option)   コンテナを起動する。Dockerfile等に変更があれば再ビルドして反映する。"
      echo "  -b, --build    キャッシュを利用してイメージをビルドし、コンテナを再作成・起動する"
      echo "  -f, --force   既存コンテナを削除し、キャッシュを使わずに強制的に再構築する。"
      exit 0 ;;
    *)
      echo "[ERROR] 不正なオプション: $1"
      exit 1 ;;
  esac
done

if $FORCE_FLAG; then
  echo "[INFO] 既存コンテナを停止・削除し、キャッシュを使わずに強制的に再構築します。"
  docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
  docker compose -f "$COMPOSE_FILE" build --no-cache
  docker compose -f "$COMPOSE_FILE" up -d
elif $BUILD_FLAG; then
  echo "[INFO] キャッシュを利用してイメージをビルドし、コンテナを再作成・起動します。"
  docker compose -f "$COMPOSE_FILE" build 
  docker compose -f "$COMPOSE_FILE" up -d
else
  echo "[INFO] コンテナを起動します (変更があれば再ビルドされます)。"
  docker compose -f "$COMPOSE_FILE" up -d
fi

# bash シェルに入る
#echo "[INFO] ${SERVICE_NAME} に /bin/bash で接続します。"
#docker exec -it "${SERVICE_NAME}" /bin/bash