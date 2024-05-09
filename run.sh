export PYTHONUNBUFFERED="true"

python -m cProfile -o exp.profile experiment.py --env hopper --dataset expert | tee "hopper-medium.log"
# --max_iters 1 --num_steps_per_iter 1 --num_eval_episodes 1
# snakeviz exp.profile



# for GYM_ENV in "hopper" "halfcheetah" "walker2d"; do
#     for DATASET in "medium" "medium-replay" "expert"; do
#         python experiment.py --env ${GYM_ENV} --dataset ${DATASET} --model_type dt | tee "${GYM_ENV}-${DATASET}.log"
#     done
# done