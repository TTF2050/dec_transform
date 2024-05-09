#!/bin/bash


python -m venv .env 
source .env/bin/activate
pip install gymnasium torch "tensorflow==2.15" carla wandb "cython<3" mujoco \
     transformers atari-py opencv-python blosc dopamine-rl dopaminekit \
     tensorflow-estimator matplotlib scikit-learn tf-agents \
     ray[all]


git clone https://github.com/TTF2050/d4rl.git && cd d4rl && git checkout update-to-gymnasium  && pip install -e .
python -c "import d4rl"
cd ..

git clone https://github.com/karpathy/minGPT.git && cd minGPT && pip install -e .
cd ..

python -m atari_py.import_roms /Atari-2600-VCS-ROM-Collection 