import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Conv1D, Add, Masking, Concatenate, Reshape, Permute, Dropout

import transformers


from decision_transformer.models.trajejctory_model import TrajectoryModel


class SkipConnectWrapper(tf.keras.layers.Layer):
    def __init__(self, inner_layer, *args, **kwargs):
        super().__init__()
        self.inner_layer = inner_layer
        self.adder = Add()
        self.supports_masking = True

    def call(self, x):
        # print(f'SkipConnectWrapper.call()')
        y = self.inner_layer(x)
        return self.adder([x, y])
    

class GPT2Conv1D(tf.keras.layers.Layer):
    def __init__(self, n_out, n_in, config, **kwargs):
            super().__init__(**kwargs)
            self.kernel = self.add_weight(shape=(n_in,n_out), 
                                          initialzer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range, seed=None), name='weight')
            self.bias = self.add_weight(shape=(n_out), initializer="zeros", name='bias')

    def call(self, x):
        x = tf.matmul(x, self.weight) + self.bias
        return x


class GPT2SelfAttention(tf.keras.layers.Layer):

    def __init__(self, nx, n_positions, config, scale=False, **kwargs):
        super().__init__(**kwargs)

        assert nx % config.n_head == 0
        # print(f'GPT2SelfAttention.__init__() nx {nx} | n_positions {n_positions}')

        self.base_mask = np.tril( np.ones((n_positions, n_positions),dtype=np.bool_) )
        self.mask_bias = tf.constant(-1e4)
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.c_attn = Dense(nx*3, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range))
        self.c_proj = Dense(nx, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range))
        self.attn_dropout = Dropout(config.attn_pdrop)
        self.resid_dropout = Dropout(config.resid_pdrop)

        
        # print(f'GPT2Attention.__init__() self.atten_mask.shape {self.base_mask.shape}')

    def compute_attention(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False, training=None):
        # print(f'Attention.compute_attention() split query.size {q.shape} | key.size {k.shape} | value.size {v.shape}')
        w = tf.matmul(q,k)
        if self.scale:
            w = w/tf.sqrt(tf.cast(v.shape[-1],dtype=tf.float32))
        nd, ns = w.shape[-2], w.shape[-1]

        # print(f'GPT2Attention.compute_attention() causal mask')
        mask = self.base_mask[ns-nd:ns, :ns]
        w = tf.where(mask, w, self.mask_bias)
        # print(f'GPT2Attention.compute_attention() w.shape {w.shape}')

        if attention_mask is not None:
            # print(f'GPT2Attention.compute_attention() applying attention mask {attention_mask.shape}')
            attention_mask = tf.where(attention_mask, 0.0, -10000.0)
            attention_mask = tf.reshape(attention_mask,shape=(-1,1,1,attention_mask.shape[-1]))
            # print(f'GPT2Attention.compute_attention() applying attention mask {attention_mask.shape}')
            w = w + attention_mask

        w = tf.keras.activations.softmax(w)

        w = self.attn_dropout(w, training=training)

        return tf.matmul(w,v)

    
    def merge_heads(self, x):
        x = tf.transpose(x, perm=(0,2,1,3))
        x_shape = x.shape
        return tf.reshape(x, shape=(*x_shape[:-2], x_shape[-2]*x_shape[-1]))
    

    def split_heads(self, x, is_keys=False):
        x_shape = x.shape
        x = tf.reshape(x, shape=(*x_shape[:-1], self.n_head, x_shape[-1]//self.n_head))
        if is_keys:
            return tf.transpose(x, perm=(0,2,3,1))
        else:
            return tf.transpose(x, perm=(0,2,1,3))



    # @tf.function
    def call(self, x, attention_mask, training=None):
        # print(f'GPT2Attention.call() input x.shape {x.shape}')
        x = self.c_attn(x)
        # print(f'GPT2Attention.call() c_attn x.shape {x.shape}')
        # TODO: figoure out split into kqv (not headsplit)
        Q, K, V = tf.split(x, 3, axis=-1)
        # print(f'Attention.forward() split query.size {Q.shape} | key.size {K.shape} | value.size {V.shape}')

        Q = self.split_heads(Q)
        K = self.split_heads(K, is_keys=True)
        V = self.split_heads(V)

        attn_score = self.compute_attention(Q, K, V, attention_mask)

        Z = self.merge_heads(attn_score)
        Z = self.c_proj(Z)
        Z = self.resid_dropout(Z, training=training)
        return Z



class GPT2MLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config):
        super().__init__()
        # print(f'GPT2MLP.__init__({n_embed}, {n_state})')
        n_embed = config.n_embd
        #TODO: convert to convolution
        self.conv1 = Dense(n_state, activation=config.activation_function, name='conv1', 
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range))
        self.conv2 = Dense(n_embed, name='conv2',
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range))
        self.dropout = Dropout(config.resid_pdrop)
        self.supports_masking = True

    def call(self, x, training=None, mask=None):
        # print(f'GPT2MLP.call()')
        # print(f'x.shape {x.shape}')
        # print(f'conv1 {self.conv1.__dict__}')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        # print(f'return.shape {x.shape}')
        return x


class GPT2Block(tf.keras.layers.Layer):
    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.layer_norm_1 = LayerNormalization(epsilon=config.layer_norm_epsilon)
        #TODO: validate config and args (key dim, value dim?)
        # self.mha = tf.keras.layers.MultiHeadAttention(
        #     num_heads=config.n_head,
        #     key_dim=config.n_embd,
        #     dropout=config.attn_pdrop,
        #     kernel_initializer=tf.keras.initializers.RandomNormal(
        #                        mean=0.0, stddev=config.initializer_range, seed=None
        #                        ))
        
        self.attention = GPT2SelfAttention(hidden_size, config.n_positions, config, True)
        #TODO: verify residual dropout layer?
       
        
        # skip1 = SkipConnectWrapper(Sequential([layer_norm_1, mha]))
        self.layer_norm_2 = LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)
        # skip2 = SkipConnectWrapper(Sequential([layer_norm_2, ffn]))
        self.adder = Add()

        # self.block_stack = Sequential([skip1, skip2])
        self.supports_masking = True


    def call(self, x, training=None, mask=None):
        # print(f'GPT2Block.call() x.shape {x.shape}')
        # print(f'GPT2Block.call() x mask {hasattr(x,"_keras_mask")}')
        # print(f'GPT2Block.call() mask {None if mask is None else mask.shape}')
        # print(f'GPT2Block.call() x {x}')
        
        x_norm = self.layer_norm_1(x, training=training)

        # print(f'GPT2Block.call() x_norm.shape {x_norm.shape}')
        # print(f'GPT2Block.call() x_norm mask {hasattr(x_norm,"_keras_mask")}')
        # print(f'GPT2Block.call() x_norm._keras_mask.shape {x_norm._keras_mask.shape}')

        # print(f'GPT2Block.call() mask is {mask.dtype}')
        # assert tf.reduce_all(x_norm._keras_mask == mask), "encoding mask and supplied attention mask did not match"

        # y = self.mha(query=x_norm,
        #     value=x_norm,
        #     key=x_norm,
        #     # attention_mask=mask,
        #     use_causal_mask = True,
        #     training=training)
        #TODO: verify mask
        y = self.attention(x_norm, mask, training=training)
        x = self.adder([x,y])
        y = self.mlp(self.layer_norm_2(x, training=training), training=training, mask=mask)
        return self.adder([x,y])
        
        # return self.block_stack(x)


class GPT2Stack(tf.keras.Model):
    def __init__(self, config, **kwargs ):
        super().__init__()
        # print(f'gpt2stack with {num_layers} layers')
        self.decoder_blocks = [
            GPT2Block(config.n_positions, config, scale=True)
                for _ in range(config.n_layer)
            ]
        self.supports_masking = True
        

    def call(self, x, training=None, mask=None):
        # print(f'GPT2Stack.call() x.shape {x.shape}')
        # print(f'GPT2Stack.call() input mask {hasattr(x,"_keras_mask")}')
        # print(f'GPT2Stack.call() mask {None if mask is None else mask.shape}')
        
        for block in self.decoder_blocks:
            x = block(x,training,mask)

        return x
    
class ApplyMask(tf.keras.layers.Layer):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # print(f'ApplyMask call() inputs.shape {inputs.shape} mask.shape {mask.shape}')
        if mask is None:
            return inputs

        # assert tf.rank(inputs) >= tf.rank(mask), "Rank of input must be equal or greater than mask shape"
        # assert inputs.shape[:tf.rank(mask)] == mask.shape, "mask shape must match inputs shape to the rank of the mask"
        # if inputs.shape != mask.shape:
        #     mask = tf.broadcast_to(tf.expand_dims(mask, axis=-1), inputs.shape)
        inputs._keras_mask = mask
        #TODO: set masked elements to zero?
        return inputs


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        print(config)

        
        # Embedding accepts inputs of a max of 
        self.mask_generator = Masking(-1)
        self.apply_mask = ApplyMask()
        self.adder = Add()
        self.embed_timestep = Embedding(max_ep_len+1, hidden_size, mask_zero=True)
        self.embed_return = Dense(hidden_size, name='embed_return', kernel_initializer='he_uniform')
        self.embed_state = Dense(hidden_size, name='embed_state', kernel_initializer='he_uniform')
        self.embed_action = Dense(hidden_size, name='embed_action', kernel_initializer='he_uniform')

        # self.reshape1 = Reshape((20, 1, self.hidden_size))
        # self.concat = Concatenate(axis=1)
        # self.permute = Permute((2, 1, 3))
        # self.reshape2 = Reshape((-1, 3*20, self.hidden_size))
        
        # Double normalization?
        self.embed_ln = LayerNormalization(epsilon=config.layer_norm_epsilon)

        # from GPTModel
        self.dropout = Dropout(config.embd_pdrop)
        self.GPT2_stack = GPT2Stack(config)
        #Matches reference GPT2Model.ln_f
        self.final_ln = LayerNormalization(epsilon=config.layer_norm_epsilon)

        # note: we don't predict states or returns for the paper
        # tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.initializer_range, seed=None)
        self.predict_state = Dense(self.state_dim, name='predict_state', kernel_initializer='he_uniform')
        self.predict_action = Dense(self.act_dim, activation='tanh' if action_tanh else '', name='predict_action', kernel_initializer='he_uniform') 
        self.predict_return = Dense(1, name='predict_return', kernel_initializer='he_uniform')

    # @tf.function
    # def forward(self, states, actions, rewards, returns_to_go, timesteps, validity_mask=None):

    def call(self, x, training=None):
        states, actions, rewards, returns_to_go, timesteps, validity_mask = x
    

        batch_size, seq_length = states.shape[0], states.shape[1]
        # print(f' > forward() batch_size: {batch_size} | seq_len {seq_length}')
        # print(f'forward() states.shape {states.shape}')
        # print(f'forward() actions.shape {actions.shape}')
        # print(f'forward() returns_to_go.shape {returns_to_go.shape}')

        
        # embed each modality with a different head
        # print(f'forward() timesteps.shape {timesteps.shape}')
        time_embeddings = self.embed_timestep(timesteps)
        # print(time_embeddings)
        # print(f'forward() time_embeddings.shape {time_embeddings.shape}')
        # print(f'forward() time_embeddings has mask: {hasattr(time_embeddings,"_keras_mask")}')
        time_mask = time_embeddings._keras_mask
        # print(f'forward() time_embeddings._keras_mask.shape {time_embeddings._keras_mask.shape}')

        # assert tf.reduce_all(validity_mask == time_mask), f"masks differ {validity_mask} vs {time_mask} - {timesteps}"
        
        # NOTE: reference does not zero the pad values. this seems... wrong
        states = self.apply_mask(states, time_mask)
        actions = self.apply_mask(actions, time_mask)
        returns_to_go = self.apply_mask(returns_to_go, time_mask)
        
        # print(f'states {states}')
        state_embeddings = self.embed_state(states)
        # print(f'state_embeddings {state_embeddings}')
        # print(f'state_embeddings.shape {state_embeddings.shape}')
        state_embeddings = self.adder((state_embeddings, time_embeddings))
        # print(f'state_embeddings + time_embeddings {state_embeddings}')
        # print(f'forward() state_embeddings.shape {state_embeddings.shape}')
        # print(f'forward() state_embeddings has mask: {hasattr(state_embeddings,"_keras_mask")}')
        
        # print(f'actions {actions}')
        action_embeddings = self.embed_action(actions)
        # print(f'action_embeddings {action_embeddings}')
        # print(f'action_embeddings.shape {action_embeddings.shape}')
        action_embeddings = self.adder((action_embeddings, time_embeddings))
        # print(f'action_embeddings + time_embeddings {action_embeddings}')
        # print(f'forward() action_embeddings.shape {action_embeddings.shape}')
        # print(f'forward() action_embeddings has mask: {hasattr(action_embeddings,"_keras_mask")}')
        
        # print(f'returns_to_go {returns_to_go}')
        returns_embeddings = self.embed_return(returns_to_go)
        # print(f'returns_embeddings {returns_embeddings}')
        # print(f'returns_embeddings.shape {returns_embeddings.shape}')
        returns_embeddings = self.adder((returns_embeddings, time_embeddings))
        # print(f'returns_embeddings + time_embeddings {returns_embeddings}')
        # print(f'forward() returns_embeddings.shape {returns_embeddings.shape}')
        # print(f'forward() returns_embeddings has mask: {hasattr(returns_embeddings,"_keras_mask")}')


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # also this operation  removes all masks
        stacked_inputs = tf.reshape(
                tf.transpose(
                    tf.stack(
                        (returns_embeddings, state_embeddings, action_embeddings), 
                        axis=1), 
                    perm=(0, 2, 1, 3)), 
                (batch_size, 3*seq_length, self.hidden_size)
            )
        # print(f'forward() stacked_inputs.shape {stacked_inputs.shape}')
        # print(f'forward() stacked_inputs has mask: {hasattr(stacked_inputs,"_keras_mask")}')

        # print(f'forward() stacked_inputs {stacked_inputs}')

        # print(f'return {returns_embeddings[0,:5,0]}')
        # print(f'state {state_embeddings[0,:5,0]}')
        # print(f'action {action_embeddings[0,:5,0]}')
        # print(f'stacked_inputs {stacked_inputs[0,:15,0]}')
        #NOTE: restore
        stacked_inputs = self.embed_ln(stacked_inputs, training=training)
        # print(f'forward() embedded stacked_inputs {stacked_inputs}')
        
        if validity_mask is None:
            # print(f'forward() ---  received no validity_mask')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            validity_mask = tf.ones((batch_size, seq_length), dtype=tf.float32)



        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_validity_mask = tf.reshape(
                tf.transpose(
                    tf.stack(
                        (validity_mask, validity_mask, validity_mask), 
                        axis=1),
                    perm=(0, 2, 1)),
                (batch_size, 3*seq_length)
            )
        # stacked_validity_mask = tf.expand_dims(stacked_validity_mask, axis=-1)
        # print(f'forward() stacked_validity_mask.shape {stacked_validity_mask.shape}')
        # we feed in the input embeddings (not word indices as in NLP) to the model


        #apply the mask to the stacked inputs before proceeding
        stacked_inputs = self.apply_mask(stacked_inputs, stacked_validity_mask)

        ## GPT2 core
        #NOTE: restore
        stacked_inputs = self.dropout(stacked_inputs, training=training)

        transformer_outputs = self.GPT2_stack(
            stacked_inputs, training
        )
        #NOTE: restore
        transformer_outputs = self.final_ln(transformer_outputs, training=training)
        # print(f'forward() output.shape {transformer_outputs.shape}')
        x = transformer_outputs #['last_hidden_state']
        

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        x = tf.transpose(
                tf.reshape(x, (batch_size, seq_length, 3, self.hidden_size))
            , perm=(0, 2, 1, 3)
            )
        # print(f'forward() reshaped transformer_outputs {x}')
        # get predictions
        #TODO: apply mask?
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        # print(f'forward() return_preds {return_preds}')
        # print(f'forward() state_preds {state_preds}')
        # print(f'forward() action_preds {action_preds}')

        # assert False
        return state_preds, action_preds, return_preds

    # @tf.function
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        
        
        if self.max_length is not None:
            # clip to len
            states = states[-self.max_length:,:]
            actions = actions[-self.max_length:,:]
            returns_to_go = returns_to_go[-self.max_length:]
            timesteps = timesteps[-self.max_length:]

            # left pad if necessary
            # assert timesteps.shape[0] == states.shape[0]
            # assert timesteps.shape[0] == actions.shape[0]
            # assert timesteps.shape[0] == returns_to_go.shape[0]
            
            seqlen = timesteps.shape[0]
            validity_mask = tf.concat([tf.zeros(self.max_length-seqlen,dtype=tf.bool), tf.ones(seqlen,dtype=tf.bool)],0)

            states = tf.concat([tf.zeros((self.max_length-seqlen, self.state_dim)), states],0)
            actions = tf.concat([tf.zeros((self.max_length-seqlen, self.act_dim)), actions],0)
            returns_to_go = tf.concat([tf.zeros(self.max_length-seqlen), returns_to_go],0)
            timesteps = tf.concat([tf.zeros(self.max_length-seqlen), timesteps],0)
            # assert timesteps.shape[0] == states.shape[0]
            # assert timesteps.shape[0] == actions.shape[0]
            # assert timesteps.shape[0] == returns_to_go.shape[0]

            states = tf.expand_dims(states,axis=0)
            actions = tf.expand_dims(actions,axis=0)
            returns_to_go = tf.expand_dims(returns_to_go,axis=0)
            returns_to_go = tf.expand_dims(returns_to_go,axis=-1)
            timesteps = tf.expand_dims(timesteps,axis=0)
            validity_mask = tf.expand_dims(validity_mask,axis=0)

        else:
            validity_mask = None
        # print("model call")
        x = (states, actions, None, returns_to_go, timesteps, validity_mask)
        _, action_preds, _ = self(x, training=False)

        return action_preds[0,-1]

    def get_batch_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        
        if self.max_length is not None:
            # clip to len
            states = states[:,-self.max_length:,:]
            actions = actions[:,-self.max_length:,:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[-self.max_length:]

            # left pad if necessary
            # assert timesteps.shape[0] == states.shape[0]
            # assert timesteps.shape[0] == actions.shape[0]
            # assert timesteps.shape[0] == returns_to_go.shape[0]
            batch_size = states.shape[0]
            seqlen = states.shape[1]
            validity_mask = tf.concat([tf.zeros((batch_size, self.max_length-seqlen), dtype=tf.bool), tf.ones((batch_size, seqlen), dtype=tf.bool)], 1)

            states = tf.concat([tf.zeros((batch_size, self.max_length-seqlen, self.state_dim)), states], axis=1)
            actions = tf.concat([tf.zeros((batch_size, self.max_length-seqlen, self.act_dim)), actions], 1)
            returns_to_go = tf.concat([tf.zeros((batch_size, self.max_length-seqlen)), returns_to_go], 1)
            timesteps = tf.concat([tf.zeros(self.max_length-seqlen), timesteps], 0)
            # assert timesteps.shape[0] == states.shape[0]
            # assert timesteps.shape[0] == actions.shape[0]
            # assert timesteps.shape[0] == returns_to_go.shape[0]

            # states = tf.expand_dims(states,axis=0)
            # actions = tf.expand_dims(actions,axis=0)
            # returns_to_go = tf.expand_dims(returns_to_go,axis=0)
            returns_to_go = tf.expand_dims(returns_to_go, axis=-1)
            # timesteps = tf.expand_dims(timesteps,axis=0)
            # validity_mask = tf.expand_dims(validity_mask,axis=0)

        else:
            validity_mask = None
        x = (states, actions, None, returns_to_go, timesteps, validity_mask)
        _, action_preds, _ = self(x, training=False)

        return action_preds[:,-1,:]