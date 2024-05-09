import gym
import numpy as np
import tensorflow as tf
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc_model import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories import StepType

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class WarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_learning_rate, warmup_steps, *args, **kwargs):
        self.target_learning_rate = target_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return tf.minimum((step+1)/self.warmup_steps, 1)*self.target_learning_rate

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
        gym_env_id = 'Hopper-v3'
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
        gym_env_id = 'HalfCheetah-v3'
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
        gym_env_id = 'Walker2d-v3'
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    # print(f'trajectories len {len(trajectories)}')
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, traj_total_return = [], [], []
    for path in trajectories:
        # print(len(path['observations']))
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        traj_total_return.append(path['rewards'].sum())
        # reshape data for later...
        if 'terminals' in path:
            path['dones'] = path['terminals']
        path['rtg'] = discount_cumsum(path['rewards'], gamma=1.)
    traj_lens, traj_total_return = np.array(traj_lens), np.array(traj_total_return)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    # print(f'states.shape {states.shape}')
    # print(f'returns.shape {traj_total_return.shape}')
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(traj_total_return):.2f}, std: {np.std(traj_total_return):.2f}')
    print(f'Max return: {np.max(traj_total_return):.2f}, min: {np.min(traj_total_return):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_return_inds = np.argsort(traj_total_return)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_return_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_return_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_return_inds[ind]]
        num_trajectories += 1
        ind -= 1
    # clip the index list down to just the selected ons
    sorted_return_inds = sorted_return_inds[-num_trajectories:]
    # print(sorted_return_inds)
    # print(traj_total_return[sorted_return_inds])
    # print(traj_lens[sorted_return_inds])

    # used to reweight sampling so we sample according to trajectory length
    p_sample = traj_lens[sorted_return_inds] / sum(traj_lens[sorted_return_inds])

    def get_batch(batch_size=256, max_len=K):
        # should really do this the easy way... tfds.load('d4rl_mujoco_hopper/v2-medium')
        """
        Generates a batch of data from the loaded offline data. 

        Creates 'batch_size' sequences of length 'max_len'.

        Returns a tuple of (state, action, reward, done, return to go, timesteps, mask)
        
        
        """
        # print(f'get_batch({batch_size}, {max_len}) from {num_trajectories} trajectories')
        # this selects batch_size indices from all trajectories, weighted by trajectory length
        batch_source_indices = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to length
        )
        # NOTE:remove
        # batch_source_indices = np.zeros(shape=batch_size)

        # s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        # precompute some useful data
        batch_sources = [trajectories[sorted_return_inds[int(idx)]] for idx in batch_source_indices]
        # print(len(batch_sources))
        batch_lens = [traj['rewards'].shape[0] for traj in batch_sources]
        # print(len(batch_lens))

        start_indices = [random.randint(0, batch_len - 1) for batch_len in batch_lens]
        # NOTE:remove
        # start_indices = [0 for _ in batch_lens]
        # print(len(start_indices))
        end_indices = list(map(lambda x,y: min(x+max_len, y), start_indices, batch_lens))
        # print(len(end_indices))
        seq_lens = [end_idx-start_idx for start_idx, end_idx in zip(start_indices, end_indices)]
        # print(len(seq_lens))
        # compute the required outputs (variable length at this stage)
        timesteps = [np.arange(start_idx, end_idx) for start_idx, end_idx in zip(start_indices, end_indices)]
        # print(f'prealloc memory')
        s = np.zeros((batch_size,max_len,state_dim),dtype=np.float32)
        a = np.ones((batch_size,max_len,act_dim),dtype=np.float32)*-10
        r = np.zeros((batch_size,max_len),dtype=np.float32)
        d = np.ones((batch_size,max_len),dtype=np.float32)*2
        rtg = np.zeros((batch_size,max_len),dtype=np.float32)
        mask = np.zeros((batch_size,max_len),dtype=np.bool_)
        # print('loop over trajectories')
        for i, params in enumerate(zip(batch_sources, timesteps, seq_lens)):
            traj, t_steps, seq_len = params
            # if seq_len != max_len:
                # print(' >> short trajectory ')
            s[i,-seq_len:,:] = traj['observations'][t_steps]
            a[i,-seq_len:,:] = traj['actions'][t_steps]
            r[i,-seq_len:] = traj['rewards'][t_steps]
            d[i,-seq_len:] = traj['dones'][t_steps]
            rtg[i,-seq_len:] = traj['rtg'][t_steps] / scale
            mask[i,-seq_len:] = np.ones((seq_len,),dtype=np.bool_)

        s -= state_mean
        s /= state_std
        
        r = np.expand_dims(r, axis=-1)
        d = np.expand_dims(d, axis=-1)
        rtg = np.expand_dims(rtg, axis=-1)
        # print(f'rtg.shape {rtg.shape}')

        
        # s = [traj['observations'][t_steps] + [np.zeros((max_len-seq_len,*state_dim))] for traj, t_steps, seq_len in zip(batch_sources, timesteps, seq_lens)]
        # a = [traj['actions'][t_steps] + [np.zeros_like(act_dim)]*(max_len-seq_len) for traj, t_steps, seq_len in zip(batch_sources, timesteps, seq_lens)]
        # r = [traj['rewards'][t_steps] + [np.zeros_like(1)]*(max_len-seq_len) for traj, t_steps, seq_len in zip(batch_sources, timesteps, seq_lens)]
        # # this works because 'terminals' was previously remapped to 'dones' (if necessary)
        # d = [traj['dones'][t_steps] + [np.zeros_like(1)]*(max_len-seq_len) for traj, t_steps, seq_len in zip(batch_sources, timesteps, seq_lens)]
        # # TODO: pretty sure that this is correct, and the weird +1 element in the original 
        # # implementation isnt actually used anywhere
        # rtg = [traj['rtg'][t_steps] + [np.zeros_like(1)]*(max_len-seq_len) for traj, t_steps, seq_len in zip(batch_sources, timesteps, seq_lens)]
        
        # pad out the data structures to max_len
        # only timesteps is explicitly padded in such a way as to encode mask data
        timesteps = tf.keras.utils.pad_sequences(timesteps,max_len,value=-1,padding='pre')
        # 0 is now the mask value, and timesteps are effecitively 1-indexed
        timesteps +=1
        # s = s + [np.zeros_like(state_dim)]*(max_len-seq_len)
        # a = a + [np.zeros_like(act_dim)]*(max_len-seq_len)
        # r = r + [np.zeros_like(1)]*(max_len-seq_len)
        # d = d + [np.zeros_like(1)]*(max_len-seq_len)
        # rtg = rtg + [np.zeros_like(1)]*(max_len-seq_len)

        # print(f'get_batch() s.shape {s.shape} | a.shape {a.shape} | r.shape {r.shape} | d.shape {d.shape}')
        # print(f'get_batch() timesteps {timesteps}')

        # print(f'batch mask dtype is {mask.dtype}')
        # print(f'actions dtype  is {a.dtype}')

        return s, a, r, d, rtg, timesteps, mask
        

    def eval_ep_parallel(target_rew):
        def fn(model):
            
            dones = tf.zeros((num_eval_episodes,1), dtype=tf.bool)
            wrapped_envs = [GymWrapper(gym.make(gym_env_id)) for _ in range(num_eval_episodes)]

            # placeholders (episodes, max_ep_len, param_dim)
            states = tf.zeros((num_eval_episodes,0,state_dim))
            actions = tf.zeros((num_eval_episodes,0,act_dim))
            # (episodes, max_ep_len)
            rewards = tf.zeros((num_eval_episodes,0))
            target_returns = tf.ones((num_eval_episodes,1))*target_rew/scale
            # (max_ep_len)
            timesteps = tf.ones((1,)) #1-indexed

            ts_s = [wrapped_env.reset() for wrapped_env in wrapped_envs]
            new_states = [tf.cast(ts.observation, dtype=tf.float32) for ts in ts_s]
            # dims (episodes, param_dim)
            new_states = tf.convert_to_tensor(new_states)
            if mode == 'noise':
                new_states = new_states + tf.random.normal(0, 0.1, size=new_states.shape)
            states = tf.concat([states, tf.expand_dims(new_states, axis=1)], axis=1)
            
            episode_return = tf.zeros(num_eval_episodes)
            episode_length = tf.zeros(num_eval_episodes, dtype=tf.int32)
            for t in range(max_ep_len):
                #tmp pad action and reward
                actions = tf.concat([actions, tf.zeros((num_eval_episodes,1,act_dim))], axis=1)
                rewards = tf.concat([rewards, tf.zeros((num_eval_episodes,1))], axis=1)
                # should be (episodes, 1, param_dim)
                new_actions = model.get_batch_action(
                    (states - state_mean) / state_std,
                    actions,
                    rewards,
                    target_returns,
                    timesteps,
                )
                # pop and replace the padding action for the real action
                actions = tf.concat([actions[:,:-1,:], tf.expand_dims(new_actions, axis=1)], axis=1)

                ts_s = [env.step(action) for env,action in zip(wrapped_envs,new_actions)]
                    
                new_states = tf.convert_to_tensor([tf.cast(ts.observation, dtype=tf.float32) for ts in ts_s])
                new_rewards = tf.convert_to_tensor([tf.cast(ts.reward, dtype=tf.float32) for ts in ts_s])
                new_dones = tf.convert_to_tensor([ts.step_type == StepType.LAST for ts in ts_s], dtype=tf.bool)

                new_dones = tf.reduce_any(tf.stack([dones[:,-1], new_dones], axis=1),axis=1)
                dones = tf.concat([dones[:,:-1], tf.expand_dims(new_dones, axis=-1)], axis=1)
                done = tf.reduce_all(tf.reduce_any(dones, axis=-1))

                states = tf.concat([states, tf.expand_dims(new_states, axis=1)], axis=1)
                rewards = tf.concat([rewards[:,:-1], tf.expand_dims(new_rewards, axis=1)], axis=1)

                if mode != 'delayed':
                    pred_return = target_returns[:,-1] - (new_rewards/scale)
                else:
                    pred_return = target_returns[:,-1]
                target_returns = tf.concat([target_returns, tf.expand_dims(pred_return,axis=1)], axis=1)
                
                timesteps = tf.concat([timesteps, tf.expand_dims(tf.cast(t+2,dtype=tf.float32),0)], 0)

                episode_return += tf.where(new_dones, 0., new_rewards)
                episode_length += tf.where(new_dones, 0, 1)

                if done:
                    break
            

            return {
                f'target_{target_rew}_return_mean': tf.math.reduce_mean(episode_return),
                f'target_{target_rew}_return_std': tf.math.reduce_std(episode_return),
                f'target_{target_rew}_length_mean': tf.math.reduce_mean(tf.cast(episode_length,dtype=tf.float32)),
                f'target_{target_rew}_length_std': tf.math.reduce_std(tf.cast(episode_length,dtype=tf.float32)),
            }
        return fn


    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for i in range(num_eval_episodes):
                print(f'starting eval {i}')
                #NOTE: with torch.no_grad():
                if model_type == 'dt':
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                else:
                    ret, length = evaluate_episode(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    # model = model.to(device=device)

    # warmup_steps = variant['warmup_steps']

    # model.build(input_shape=(None,20,11))
    # model.summary()

    #TODO valaidate
    scheduler = WarmupLR(
        variant['learning_rate'],
        variant['warmup_steps']
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduler,
        weight_decay=variant['weight_decay'],
        global_clipnorm=.25
    )

    @tf.function
    def loss_fn(s_hat, a_hat, r_hat, s, a, r, mask=None):
        if mask is not None:
            mask =tf.cast(mask, dtype=tf.float32)
            return tf.reduce_mean(((a_hat - a)**2)*mask)
        return tf.reduce_mean((a_hat - a)**2)
    
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[eval_ep_parallel(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: tf.reduce_mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

def configureGPUs(gpu_id=None, mem_limit=None):
    """Sets memory limits on one or all GPUs"""
    #get list of all GPUs in the system
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #if no specific GPU is specified, iterate over all GPUs
    if gpu_id is None:
        for gpu in gpus:
            #tf.config.experimental.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            if mem_limit is not None:
                tf.config.experimental.set_virtual_device_configuration(gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
    else:
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        if mem_limit is not None:
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--mem_limit', type=int, default=768)
    
    args = parser.parse_args()

    configureGPUs(mem_limit=args.mem_limit)

    experiment('gym-experiment', variant=vars(args))
