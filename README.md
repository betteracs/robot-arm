## Installation

```
conda create -n robot-proj python=3.11 -y 
conda activate robot-proj
pip install uv
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

cd LIBERO
pip install -e .
cd ..
uv pip install -r requirements.txt
python download_datasets.py
```