# borrowed from VILA install_env.sh
conda create -n timerefine python=3.10 -y # make sure you install python 3.10
conda activate timerefine 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# change torch, torchvision to the correct version

pip install --upgrade pip  # enable PEP 660 support
# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda-toolkit -y

pip install -r requirements.txt

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install peft==0.10.0
pip install git+https://github.com/huggingface/transformers@v4.36.2
