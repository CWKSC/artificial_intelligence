python -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip
pip install matplotlib
pip install pandas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchdata
pip install torchtext
pip install spacy
pip install portalocker
pip install seaborn

./install.ps1

pause
deactivate