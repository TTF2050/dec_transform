## for reference only
exit 1

WORKDIR="{HOME}/workdir"

## initial setup on host/login node
mkdir "${WORKDIR}"
cd "${WORKDIR}"
#read-only clone and checkout
git clone https://github.com/TTF2050/dec_transformer.git 

#build image
singularity build dectransform.simg docker://ttf2050/dectransform:0.0.2
#start image
singularity run dectransform.simg
#put the env configuration script in the right place
cd "${WORKDIR}"
cp decision-transformer/bootstrap_env.sh .
./bootstrap_env.sh
source .env/bin/activate
cd decision-transformer/gym/data
python download_d4rl_datasets.py

#prime the grid launcher
cd ~
cp "${WORKDIR}"/decision-transformer/gym/run_grid.sh .
cp "${WORKDIR}"/decision-transformer/gym/oscar_job.sh .
# exit image shell ctrl+d


## do stuff
interact -q gpu -g 2
singularity run --nv dectransform.simg
cd "${WORKDIR}"/
source .env/bin/activate
cd decision-transformer/gym
./run_grid.sh
