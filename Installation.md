```
conda create -n robot-proj python=3.11 -y 
conda activate robot-proj
pip install uv
git clone https://github.com/huggingface/lerobot-libero
cd lerobot-libero
uv pip install numpy==2.4.4
uv pip install -r requirements.txt
uv pip install -e .
uv pip install num2words
uv pip install -u robotsuite_models
python /home/shinawatra/miniconda3/envs/robot-proj/lib/python3.11/site-packages/robosuite/scripts/setup_macros.py
```