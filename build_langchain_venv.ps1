py -3.9 -m venv venv_langchain
.\venv_langchain\Scripts\activate

python -m pip install --upgrade pip

pip install langchain[llms]
pip install transformers accelerate einops xformers bitsandbytes sentence_transformers

