py -3.9 -m venv venv_tensorflow
.\venv_tensorflow\Scripts\activate

python -m pip install --upgrade pip

./install_common.ps1

pip install tensorflow
pip install protobuf=4.20
pip install tensorflow_datasets
pip install tensorflow-text

./install.ps1

pause
deactivate