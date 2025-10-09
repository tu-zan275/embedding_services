#!/bin/bash

set -e

echo "=== Khởi động Milvus ==="
docker-compose up -d

echo "=== Hoàn tất! ==="
echo "Milvus đang chạy tại:"
echo " - gRPC: localhost:19530"
echo " - REST: localhost:9091"
