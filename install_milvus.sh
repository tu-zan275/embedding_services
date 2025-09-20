#!/bin/bash

set -e

echo "=== Tạo thư mục milvus ==="
mkdir -p ~/milvus && cd ~/milvus

echo "=== Tạo docker-compose.yml ==="
cat > docker-compose.yml << 'EOF'
version: '3.5'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379
    ports:
      - "2379:2379"

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-02-09T05-16-53Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    ports:
      - "9000:9000"

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"   # gRPC
      - "9091:9091"     # RESTful API
    depends_on:
      - "etcd"
      - "minio"
EOF

echo "=== Khởi động Milvus ==="
docker-compose up -d

echo "=== Hoàn tất! ==="
echo "Milvus đang chạy tại:"
echo " - gRPC: localhost:19530"
echo " - REST: localhost:9091"
