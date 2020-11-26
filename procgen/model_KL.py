import tensorflow as tf
import functools

from procgen.tf_util import get_session, save_variables, load_variables_new
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
        self.R = R = tf.placeholder(tf.float32, [None])
          # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        # Get the predicted value
        vpred = train_model.new_vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.new_vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        #vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        vf_loss = tf.losses.mean_squared_error(tf.stop_gradient(train_model.vf),train_model.new_vf)
        # reset loss // reset version
        
        pi_kl_loss = tf.reduce_mean(train_model.new_pd.kl(train_model.pd))
        reset_loss = pi_kl_loss + vf_loss
        
        # Total loss
        
        
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters // reset version
        total_params = tf.trainable_variables('ppo2_model')
        params = [v for v in total_params if ('new_impala' in v.name) and ('Adam' not in v.name)] #srat
        l2_weight = tf.reduce_sum([tf.nn.l2_loss(v) for v in params])

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
      
        self.reset_loss_names = ['l2_loss','vf_loss', 'impala_network_size']
        self.reset_stats_list = [reset_loss, vf_loss, l2_weight] #reset
     

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.new_step = act_model.new_step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
     
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables_new, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
    
   
    def reset_train(self,lr,cliprange, obs, returns, masks,values, states=None):
       
        td_map = {
            self.train_model.X : obs,
            self.R : returns,
            self.CLIPRANGE : cliprange,
            self.LR : lr,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.reset_stats_list + [self._reset_train_op], # reset
            td_map
        )[:-1]
        
    def reset_grad(self,lr,cliprange, obs, returns, masks,values, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
       
        td_map = {
            self.train_model.X : obs,
            self.R : returns,
            self.CLIPRANGE : cliprange,
            self.LR : lr,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.reset_grads_norm,
            td_map
        )