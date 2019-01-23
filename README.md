Anonymous Codebase for Collaborative Evolutionary Reinforcement Learning for submission to ICML 2019

#################################
          Code labels
#################################

main.py: Main Script runs everything

core/runner.py: Rollout worker

core/ucb.py: Upper Confidence Bound implemented for learner selection by the resource-manager

core/portfolio.py: Portfolio of learners which can vary in their hyperparameters

core/learner.py: Learner agent encapsulating the algo and sum-statistics

core/buffer.py: Cyclic Replay buffer

core/env_wrapper.py: Wrapper around the Mujoco env

core/models.py: Actor/Critic model

core/neuroevolution.py: Implements Neuroevolution

core/off_policy_algo.py: Implements the off_policy_gradient learner TD3

core/mod_utils.py: Helper functions



######################
  REPRODUCE RESULTS
######################

python main.py -env HalfCheetah-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Hopper-v2 -portfolio {10,14} -total_steps 1.5 -seed {2018,2022}

python main.py -env Humanoid-v2 -portfolio {10,14} -total_steps 1 -seed {2018,2022}

python main.py -env Walker2d-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Swimmer-v2 -portfolio {10,14} -total_steps 2 -seed {2018,2022}

python main.py -env Hopper-v2 -portfolio {100,102} -total_steps 5 -seed {2018,2022}

where {} represents an inclusive discrete range: {10, 14} --> {10, 11, 12, 13, 14}


######################
        NOTE
######################

All roll-outs (evaluation of actors in the evolutionary population and the explorative roll-outs 
conducted by the learners run in parallel). They are farmed out to different CPU cores, 
and write asynchronously to the collective replay buffer. Thus, slight variations in results 
are observed even with the same seed. 
