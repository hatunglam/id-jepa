#!/bin/bash
set -e  # Exit on error

PYTHON_VERSION=3.10.14
VENV_NAME=venv310
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # The directory where the script is located

echo ">>> Installing required build tools..."
sudo yum install -y gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel xz xz-devel libffi-devel wget make

echo ">>> Downloading and building Python $PYTHON_VERSION..."
cd /usr/src
if [ ! -f "Python-$PYTHON_VERSION.tgz" ]; then
    sudo wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
fi
sudo tar xzf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
sudo ./configure --enable-optimizations
sudo make altinstall

echo ">>> Returning to project directory..."
cd "$SCRIPT_DIR"

echo ">>> Creating virtual environment ($VENV_NAME)..."
/usr/local/bin/python3.10 -m venv "$VENV_NAME"

echo ">>> Activating virtual environment..."
source "$VENV_NAME/bin/activate"

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠ No requirements.txt found in $SCRIPT_DIR"
fi

echo "✅ Setup complete!"
python --version
