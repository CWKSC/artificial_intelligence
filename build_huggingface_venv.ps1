py -3.9 -m venv venv_huggingface
.\venv_huggingface\Scripts\activate

python -m pip install --upgrade pip

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install huggingface_hub
pip install git+https://github.com/huggingface/transformers.git

# https://huggingface.co/docs/transformers/main_classes/quantization
pip install auto-gptq
pip install git+https://github.com/huggingface/optimum.git
pip install --upgrade accelerate
