import tensorflow as tf
import functools

from procgen.tf_util import get_session, save_variables, load_variables
from procgen.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, 
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
       
        self.LR = LR = tf.placeholder(tf.float32, [])


        # CALCULATE THE LOSS


        # reset loss // reset version
        pi_l2_loss = tf.losses.mean_squared_error(tf.stop_gradient(train_model.pi), train_model.new_pi) 
        vf_l2_loss = tf.losses.mean_squared_error(tf.stop_gradient(train_model.vf),train_model.new_vf)
        reset_loss = pi_l2_loss + vf_l2_loss

	
        # Total loss
        
        
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters // reset version
        total_params = tf.trainable_variables('ppo2_model')
        params = [v for v in total_params if 'new' in v.name] #srat
       
        weight_params_cnn = [v for v in params if('layer_15' not in v.name) and ('Adam' not in v.name)]
        weight_params_impala_fc = [v for v in params if('layer_15' in v.name) and ('Adam' not in v.name)]
        weight_params_policy_fc = [v for v in params if ('new_pi' in v.name) and ('Adam' not in v.name)]
        weight_params_value_fc = [v for v in params if ('new_vf' in v.name) and ('Adam' not in v.name)]
        print(params)
        cnn_weight = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params_cnn])
        impala_fc_weight = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params_impala_fc])
        policy_fc_weight = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params_policy_fc])
        value_fc_weight = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params_value_fc])            
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in params])
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        reset_grads_and_var = self.trainer.compute_gradients(reset_loss, params) # // reset version

        reset_grads, reset_var = zip(*reset_grads_and_var)
        if max_grad_norm is not None:
            reset_grads, _reset_grad_norm = tf.clip_by_global_norm(reset_grads, max_grad_norm)
        reset_grads_and_var = list(zip(reset_grads, reset_var))
        

        self.reset_grads = reset_grads
        self.reset_grads_norm = _reset_grad_norm
        self.reset_var = reset_var

        self._reset_train_op = self.trainer.apply_gradients(reset_grads_and_var) # // reset version
      
        self.reset_loss_names = ['l2_loss', 'pi_l2_loss','vf_l2_loss', 'l2_impala_loss']
        
        self.reset_stats_list = [reset_loss, pi_l2_loss, vf_l2_loss, l2_loss] #reset
        
        self.weight_names = ['cnn', 'impala_fc', 'policy_fc', 'value_fc']
        self.weight_list = [cnn_weight,impala_fc_weight, policy_fc_weight,value_fc_weight]
  
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.new_step = act_model.new_step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
     
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
    
   
    def reset_train(self, lr, obs, masks, states=None):
       
        td_map = {
            self.train_model.X : obs,
            self.LR : lr
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.reset_stats_list + [self._reset_train_op], # reset
            td_map
        )[:-1]
    def weight(self):
        return self.sess.run(self.weight_list)    
    def reset_grad(self,lr, obs, masks, states=None ):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
       
        td_map = {
            self.train_model.X : obs,
            self.LR : lr
           
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.reset_grads_norm,
            td_map
        )

    