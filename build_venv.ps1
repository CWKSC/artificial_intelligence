py -3.9 -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install pandas
pip install tqdm
pip install seaborn

pip install scikit-learn
pip install scikit-optimize

pip install tensorflow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchdata
pip install torchtext

pip install spacy
pip install portalocker
pip install ray[tune]
pip install emoji

pip install lightgbm
pip install catboost
pip install ngboost

pip install transformers[torch]
pip install xformers
 
./install.ps1

pause
deactivate