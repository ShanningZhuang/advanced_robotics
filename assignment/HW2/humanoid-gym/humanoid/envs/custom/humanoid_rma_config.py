from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class XBotLRMACfg(XBotLCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(XBotLCfg.env):
        num_height_points = 25
        # Actor Obs 
        frame_stack = 15
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        # Critic Obs
        c_frame_stack = 3
        single_num_privileged_obs = 73 + num_height_points
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        # RMA Encoder Obs
        r_frame_stack = 15
        # TODO -------------------------------------------------------
        # Set the length of single and full rma obs
        # The full rma obs is a stack of single rma obs, stack number is r_frame_stack

        single_num_rma_obs = None
        num_rma_obs = None
        # ------------------------------------------------------------

    class terrain(XBotLCfg.terrain):
        mesh_type = 'trimesh'

    class commands(XBotLCfg.commands):
        class ranges:
            lin_vel_x = [0., 1.]   # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.] # min max [rad/s]

class XBotLRMACfgPPO(XBotLCfgPPO):
    runner_class_name = "RMARunner"
    class runner( XBotLCfgPPO.runner ):
        policy_class_name = 'ActorCriticEncoder'
        algorithm_class_name = 'RMA'
        run_name = 'rma'
        experiment_name = 'XBot_rma'

    class policy( XBotLCfgPPO.policy ):
        enc_hidden_dims = [256, 256, 256]
    
    class rma:
        num_latents = 128

class XBotLRMAAdaptationCfgPPO(XBotLRMACfgPPO):
    runner_class_name = "RMAAdaptationRunner"
    class runner( XBotLRMACfgPPO.runner ):
        load_teacher = True
        policy_class_name = 'ActorCriticAdaptation'
        algorithm_class_name = 'RMAAdaptation'
        run_name = 'adaptation'
        experiment_name = 'XBot_adaptation'

    class rma( XBotLRMACfgPPO.rma ):
        teacher_load_run = -1
        teacher_checkpoint = -1
        teacher_experiment_name = "XBot_rma"
        teacher_run_name = 'rma'