## Outline of my presentation
1. put some video about humanoid robot locomotion. And raise the question. How do they do this 仿生 motion?
- 【【登长城 逐旷野】】 https://www.bilibili.com/video/BV1SRm5YEEu8/?share_source=copy_web&vd_source=943133e4e23233711998784b578fd029
2. Add some Reinforcement Learning background. Previously, the RL reward in mujoco humanoid will not generate natural motion. (like this video https://www.youtube.com/watch?v=DaYAn8Z-bYk). To make it more natural, you will need to manually add some reward like cycle of the motion, balance and other reward term. Some reward tuning will add more than 10 reward term, make it impossible to tune.
3. This paper use imitation data to shape the reward, make it natural. Also it use a method to switch between policy to make humanoid do several motions.
4. Talk about the reward settings in deepmimic paper. And each term in the reward.
5. Talk about the policy switching in deepmimic paper. (reference recent paper like PHC)
6. Conclusion
## Reward setting of huamnoid in gymnasium as background referenced in point 2
Rewards
The total reward is: reward = healthy_reward + forward_reward - ctrl_cost - contact_cost.

healthy_reward: Every timestep that the Humanoid is alive (see definition in section “Episode End”), it gets a reward of fixed value healthy_reward (default is 
).

forward_reward: A reward for moving forward, this reward would be positive if the Humanoid moves forward (in the positive 
 direction / in the right direction). 
 
, where 
 is the displacement of the center of mass (
), 
 is the time between actions, which depends on the frame_skip parameter (default is 
), and frametime which is 
 - so the default is 
, 
 is the forward_reward_weight (default is 
).

ctrl_cost: A negative reward to penalize the Humanoid for taking actions that are too large. 
, where 
 is ctrl_cost_weight (default is 
).

contact_cost: A negative reward to penalize the Humanoid if the external contact forces are too large. 
, where 
 is contact_cost_weight (default is 
), 
 are the external contact forces (see cfrc_ext section on observation).

info contains the individual reward terms.

Note: There is a bug in the Humanoid-v4 environment that causes contact_cost to always be 0.