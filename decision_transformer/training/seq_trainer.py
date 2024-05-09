import numpy as np
import tensorflow as tf

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):
    @tf.function
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, validity_mask = self.get_batch(self.batch_size)
        
        # action_target = torch.clone(actions)
        with tf.GradientTape() as tape:
            state_preds, action_preds, reward_preds = self.model(
                (states, actions, rewards, rtg, timesteps, validity_mask)
            )

            act_dim = action_preds.shape[2]
            # action_preds = tf.reshape(action_preds, (-1, act_dim))[tf.reshape(attention_mask, (-1)) > 0]
            # actions = tf.reshape(actions, (-1, act_dim))[tf.reshape(attention_mask,(-1)) > 0]

            # print(f'action_preds.shape {action_preds.shape}')
            
            # print(f'attention_mask.shape {attention_mask.shape}')

            loss = self.loss_fn(
                None, action_preds, None,
                None, actions, None,
                # tf.expand_dims(validity_mask, axis=-1)
            )

            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            # self.optimizer.step()
            # print('compute grad')
            gradients = tape.gradient(loss, self.model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # print(f'apply grads {gradients}')
            # print(f'trainable_variables {self.model.trainable_variables}')
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            # print('grad applied')
            

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = tf.reduce_mean((action_preds-actions)**2).detach().cpu().item()

        self.diagnostics['training/action_error'] = tf.reduce_mean((action_preds-actions)**2)
        return loss
        # return loss.detach().cpu().item()
