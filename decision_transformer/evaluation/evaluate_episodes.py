import numpy as np
import tensorflow as tf
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories import StepType

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

# @tf.function
def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    wrapped_env = GymWrapper(env)

    # placeholders
    states = tf.zeros((0,state_dim))
    actions = tf.zeros((0,act_dim))
    rewards = tf.zeros(0)
    target_returns = tf.ones(1)*target_return
    timesteps = tf.ones(1) #1-indexed

    ts = wrapped_env.reset()
    state = ts.observation
    # state = env.reset()
    state = tf.cast(state, dtype=tf.float32)
    if mode == 'noise':
        state = state + tf.random.normal(0, 0.1, size=state.shape)
    states = tf.concat([states,tf.expand_dims(state,axis=0)], axis=0)
    # states[0,:] = state

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print(t)
        actions = tf.concat([actions, tf.zeros((1,act_dim))], axis=0)
        rewards = tf.concat([rewards, tf.zeros(1)], axis=0)
        # print("get action")
        action = model.get_action(
            (states - state_mean) / state_std,
            actions,
            rewards,
            target_returns,
            timesteps,
        )

        actions = tf.concat([actions[:-1],tf.expand_dims(action,axis=0)], axis=0)
        # actions[-1,:] = action
        # action = action.detach().cpu().numpy()
        # print("apply action")
        # state, reward, done, _ = env.step(action.numpy())
        ts = wrapped_env.step(action)
        state = ts.observation
        reward = ts.reward
        done = (ts.step_type == StepType.LAST)


        # print('update tensors')
        # print(f'evaluate_episode_rtg() states.shape {states.shape}')
        # print(f'evaluate_episode_rtg() state.shape {state.shape}')

        states = tf.concat([states, tf.expand_dims(tf.cast(state, dtype=tf.float32),axis=0)], axis=0)
        rewards = tf.concat([rewards[:-1],tf.expand_dims(tf.cast(reward, dtype=tf.float32),axis=0)], axis=0)
        # rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_returns[-1] - (reward/scale)
        else:
            pred_return = target_returns[-1]
        target_returns = tf.concat([target_returns, tf.expand_dims(pred_return,axis=0)], axis=0)

        timesteps = tf.concat([timesteps, tf.expand_dims(tf.cast(t+2,dtype=tf.float32),axis=0)], axis=0)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

