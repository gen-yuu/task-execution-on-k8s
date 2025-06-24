#!/bin/bash

# tasks.csv に書かれたタスクの総数 (ヘッダーを除く)
TOTAL_TASKS=30

# S3の設定
S3_BUCKET="test-bucket"
S3_ENDPOINT="192.168.0.50:9000"
S3_ACCESS_KEY="minioadmin"
S3_SECRET_KEY="Ashushu0810@"

echo "Starting batch execution of ${TOTAL_TASKS} tasks..."
echo "----------------------------------------------------"

# task_index を 0 から (TOTAL_TASKS - 1) までループさせる
for (( i=0; i<${TOTAL_TASKS}; i++ ))
do
    echo ""
    echo ">>> Running task with index: ${i}"

    # main.pyを実行
    python3 main.py --test-mode \
        --task-index ${i} \
        --s3-bucket "${S3_BUCKET}" \
        --s3-endpoint-url "${S3_ENDPOINT}" \
        --s3-access-key "${S3_ACCESS_KEY}" \
        --s3-secret-key "${S3_SECRET_KEY}"

    # もし1つのタスクが失敗したらスクリプトを停止させたい場合は、以下の行のコメントを外す
    # if [ $? -ne 0 ]; then
    #     echo "!!! Task ${i} failed. Aborting script."
    #     exit 1
    # fi
done

echo "----------------------------------------------------"
echo "All tasks have been executed."