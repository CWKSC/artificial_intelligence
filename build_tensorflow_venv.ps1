py -3.9 -m venv venv_tensorflow
.\venv_tensorflow\Scripts\activate

python -m pip install --upgrade pip

pip install h5py
pip install typing-extensions
pip install wheel

pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text
pip install protobuf~=3.20.3
pip install -q tensorflow_datasets
pip install -q -U tensorflow-text tensorflow

# ./install_common.ps1

./install_custom_library.ps1

pause
deactivate