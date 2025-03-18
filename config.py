import os

import torch

class Config:
    PPO_C = False
    PPO_CB = False
    PPO_SKL = False
    A_PPO = False
    F_PPO_BD = False

    PPO_C = True
    # PPO_CB = True
    # PPO_SKL = True
    # A_PPO = True
    # F_PPO_BD = True

    #--------------F-PPO and A-PPO specific-----
    REGULARIZE_ADVANTAGE = False
    ZETA_0 = 1
    ZETA_1 = 0

    BETA_0 = 1
    BETA_1 = 0.25
    BETA_2 = 0.25
    BETA_3 = 0.
    BETA_4 = 0.

    USE_F_DELTA = False
                        
    START_ITERATION = 17
    START_THRESH = 0.5     
    END_THRESH = 0.
    RATIO = 0.985  
    OMEGA = 0.005
    #----------------------------------------

    B_EPSILON = 1e-6  # for lambda loss
    STATIC_KL_COEFF = 1.5   # for KL penalty

    BETA_LAMBDA = 0.0
    BETA_C_PI = 0.0

    PPO_CLIP_RANGE = 0.2 # for clipped version of PPO

    KL_PEN = False  # whether to use KL penalty

    STATIC_KL_TARG = 0.000001

    if PPO_SKL:
        MODEL = f'PPO'
        KL_PEN = True
    elif PPO_C:
        MODEL = f'PPO-C'
        # BETA_C_PI = 0.75  # setting 2,3
        BETA_C_PI = 0.65  # setting 1
        KL_PEN = True
    elif PPO_CB:
        MODEL = f'PPO-Cb'
        # BETA_C_PI = 0.75  # setting 2,3
        BETA_C_PI = 0.65  # setting 1
        KL_PEN = True
        USE_F_DELTA = True
        BETA_LAMBDA = 1.0
    elif A_PPO:
        MODEL = 'A-PPO'
        REGULARIZE_ADVANTAGE = True
        BETA_0 = 1
        BETA_1 = 0.25
        BETA_2 = 0.25
        BETA_3 = 0.
        BETA_4 = 0.
    elif F_PPO_BD:
        MODEL = 'F-PPO-L'
        REGULARIZE_ADVANTAGE = True
        BETA_0 = 1
        BETA_1 = 0.
        BETA_2 = 0.
        BETA_3 = 1.12                  
        BETA_4 = 1.  


    RESULTS_DIR = './results_setting1'
    ########## Experiment Setup Parameters ##########
    EXP_DIR = os.path.join(RESULTS_DIR, MODEL)
    SAVE_DIR = os.path.join(EXP_DIR, 'models')
    EVAL_DIR = os.path.join(EXP_DIR, 'evaluation')
    # PPO model paths to evaluate
    EVAL_MODEL_PATHS = {
        f'{MODEL}_final' : os.path.join(SAVE_DIR, 'final_model'),
    }

    NUM_ENVS = 1    # Number of parallel environments to run, will break if not 1
    NUM_T_SEEDS = 5  # how many training seeds to use for each model
    LOG_FULL_EPISODES = False
    LOG_EP_FREQ = 25

    ########## Env Parameters ##########
    DELAYED_IMPACT_CLUSTER_PROBS = (
        (0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),

        # (0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),
        (0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0),   
    )

    # home mortgage probs
    DELAYED_IMPACT_SUCCESS_PROBS = (0.773, 0.804, 0.833, 0.857, 0.879, 0.898, 0.914, 0.928, 0.939, 0.949, 0.958, 0.965, 0.970, 0.975 )
    
    # lending club, generated like home mortgage probs
    # DELAYED_IMPACT_SUCCESS_PROBS = (0.506, 0.594, 0.677, 0.750, 0.812, 0.861, 0.898, 0.927, 0.948, 0.963, 0.974, 0.982, 0.987, 0.991)

    def qual_change(self, curr_x, next_x):
        max_x = len(self.DELAYED_IMPACT_SUCCESS_PROBS) - 1
        qc = next_x**3 - curr_x**3
            
        return qc / (max_x**3 - (max_x - 1)**3)

    QUAL_CHANGE = qual_change

    USE_CREDIT_DRIFT = True

    DRIFT_PROBS = (
        (0.1, 0.8, 0.1),
        (0.1, 0.8, 0.1)
        # (0.05, 0.85, 0.1),
        # (0.25, 0.7, 0.05)
    )

    PI0_PROB_A1 = 0.00 # probability baseline policy gives loan

    DIST_EST = True  # whether to use the history in the observation for ppo_wrapper_env

    NUM_GROUPS = 2
    GROUP_0_PROB = 0.5
    BANK_STARTING_CASH= 10000
    INTEREST_RATE = .1

    CLUSTER_SHIFT_INCREMENT= 0.01
    CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS
    EP_TIMESTEPS = 500

    EVAL_EP_TIMESTEPS = 10_000

    WINDOW = 300

    NUM_INDIVIDUALS = int((1 / CLUSTER_SHIFT_INCREMENT) * GROUP_0_PROB) \
        + int((1 / CLUSTER_SHIFT_INCREMENT) * (1 - GROUP_0_PROB))

    ########## PPO Train Parameters ##########

    BATCH_SIZE = 50
    N_EPOCHS = 10

    TRAIN_TIMESTEPS = 500_000  # Total train time

    LEARNING_RATE = 0.00001
    POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                        net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])])
    SAVE_FREQ = 500_000

    SEED = 2023 # initial seed

    ########## Eval Parameters ##########
    # Weights for delta bank cash and delta terms in the reward for the lending environment
    EVAL_ZETA_0 = 1
    EVAL_ZETA_1 = 0
    BURNIN = 0  # Number of steps before applying the threshold policy.



    if USE_CREDIT_DRIFT == False:
        DRIFT_PROBS = (
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0)
        )






