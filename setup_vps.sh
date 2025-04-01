#!/bin/bash

# Cấu hình VPS
VPS_IP="myvps"
VPS_USER="root"
WORK_DIR="/work"
ANACONDA_INSTALLER="Anaconda3-2024.10-1-Linux-x86_64.sh"
ANACONDA_URL="https://repo.anaconda.com/archive/$ANACONDA_INSTALLER"
ANACONDA_DIR="/opt/anaconda3"

# Load mật khẩu từ file .env
export $(grep VPS_ROOT .env | xargs)

# Copy SSH key nếu chưa có
sshpass -p "$VPS_ROOT" ssh-copy-id $VPS_USER@$VPS_IP

# Kết nối SSH và thiết lập thư mục làm việc
sshpass -p "$VPS_ROOT" ssh $VPS_USER@$VPS_IP << EOF
mkdir -p $WORK_DIR
EOF

# Sao chép các file cần thiết lên VPS
sshpass -p "$VPS_ROOT" scp *.py $VPS_USER@$VPS_IP:$WORK_DIR
sshpass -p "$VPS_ROOT" scp environment.yml $VPS_USER@$VPS_IP:$WORK_DIR
sshpass -p "$VPS_ROOT" scp .env $VPS_USER@$VPS_IP:$WORK_DIR

# Cài đặt Anaconda và thiết lập môi trường
sshpass -p "$VPS_ROOT" ssh $VPS_USER@$VPS_IP << EOF
cd $WORK_DIR
wget $ANACONDA_URL
bash $ANACONDA_INSTALLER -b -p $ANACONDA_DIR
$ANACONDA_DIR/bin/conda init
EOF
sshpass -p "$VPS_ROOT" ssh $VPS_USER@$VPS_IP << EOF
cd $WORK_DIR
$ANACONDA_DIR/bin/conda env create -f environment.yml
EOF

echo "Setup VPS hoàn tất!"
