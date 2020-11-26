import tensorflow as tf
import functools

from procgen.tf_util import get_session, save_variables, load_variables_sample
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
        
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])


        
        vf_loss = tf.losses.mean_squared_error(tf.stop_gradient(train_model.target_vf),train_model.new_vf)
        # reset loss // reset version
        pi_kl_loss = tf.reduce_mean(train_model.new_pd.kl(train_model.target_pd))
        reset_loss = pi_kl_loss + vf_loss

	
        # Total loss
        
        
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters // reset version
        total_params = tf.trainable_variables('ppo2_model')
        params = [v for v in total_params if ('new' in v.name) and ('Adam' not in v.name)] #srat
        target_params = [v for v in total_params if 'target' in v.name]
        self.target_params =  sorted(target_params, key=lambda v: v.name)
        
        sample1_params = [v for v in total_params if 'sample1' in v.name]
        self.sample1_params =  sorted(sample1_params, key=lambda v: v.name)
        sample2_params = [v for v in total_params if 'sample2' in v.name]
        self.sample2_params =  sorted(sample2_params, key=lambda v: v.name)
        sample3_params = [v for v in total_params if 'sample3' in v.name]
        self.sample3_params =  sorted(sample3_params, key=lambda v: v.name)
        sample4_params = [v for v in total_params if 'sample4' in v.name]
        self.sample4_params =  sorted(sample4_params, key=lambda v: v.name)
        sample5_params = [v for v in total_params if 'sample5' in v.name]
        self.sample5_params =  sorted(sample5_params, key=lambda v: v.name)
        
        sample_params = [v for v in total_params if 'target' not in v.name and 'new' not in v.name and 'sample' not in v.name]
        self.sample_params =  sorted(sample_params, key=lambda v: v.name)
        self.new_params = sorted(params, key=lambda v: v.name)
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
        self.reset_stats_list = [pi_kl_loss, vf_loss, l2_weight] #reset
     

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.step1 = act_model.step1
        self.step2 = act_model.step2
        self.step3 = act_model.step3
        self.step4 = act_model.step4
        self.step5 = act_model.step5
        self.new_step = act_model.new_step
        self.target_step = act_model.target_step
        self.value = act_model.value
        self.target_value = act_model.target_value
        self.initial_state = act_model.initial_state
     
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables_sample, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
    
    def copy_model_params_new(self):
        update_ops = []
        param_zip = zip(self.new_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
    
    def copy_model_params(self):
        update_ops = []
        param_zip = zip(self.target_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
    
    def copy_model_params1(self):
        update_ops = []
        param_zip = zip(self.sample1_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
    def copy_model_params2(self):
        update_ops = []
        param_zip = zip(self.sample2_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
        
    def copy_model_params3(self):
        update_ops = []
        param_zip = zip(self.sample3_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
        
    def copy_model_params4(self):
        update_ops = []
        param_zip = zip(self.sample4_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)    
     
    def copy_model_params5(self):
        update_ops = []
        param_zip = zip(self.sample5_params, self.sample_params)
    
        for e2_v, e1_v in param_zip:
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
        self.sess.run(update_ops)
        
        
    def reset_train(self,lr, obs, masks, states=None):
       
        td_map = {
            self.train_model.X : obs,           
            self.LR : lr,
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.reset_stats_list + [self._reset_train_op], # reset
            td_map
        )[:-1]