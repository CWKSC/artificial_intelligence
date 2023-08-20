py -3.9 -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip

./install_common.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchdata
pip install torchtext

./install_custom_library.ps1

pause
deactivate