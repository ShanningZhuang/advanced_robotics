# Advanced Topics in Robotics 2025: Course Project

April 15, 2025

This is the documentation for the course project of Frontier Research on Intelligent Robots 2025. You are required to finish a project related to some sub-field of robotics. We have provided some topics for you to explore, and we are also open to your own ideas beyond our proposed topic. The project weighs 35% in the whole score.

## 1 Rules

- 1-2 students per team. Each team only needs to submit one proposal and presentation slide.

- The proposal weighs 10%, due at the 11th week. The proposal should be no longer than 2 pages (excluding references) and written in NeurIPS format (final version).

- The final project presentation weighs 25%, conducted on the last week. Each team has to demonstrate your results through presentation. You should submit your slides or other materials before the last class.

## 2 Topic

Below are our suggested topics and related references, but feel free to propose your own idea if you want.

### 2.1 Locomotion

Humanoid-Gym is an easy-to-use reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment. Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujoco that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies.

Utilizing the Humanoid-Gym codebase, you can control the robot to do challenging locomotion tasks. Please turn on domain randomization and noises if you choose this topic. Here we provide some tasks for you:

1. **Control the humanoid robot to run as fast as you can.**

   The easiest way is to enlarge command domains in training. Another way is to remove commands and add linear velocity into reward terms. The above two methods are baselines and you should explore new methods. If you find PPO is sufficient to run on plane, you can try to run on various terrains.

2. **Control the humanoid robot to do long/high jump**

   You can create heuristic reward functions to guide the robot to jump. If you want high-quality reference actions you can find motion capture dataset in CMU Mocap Database and retarget to our humanoid.

3. **Walk up/down the stairs with visions**

   You can attach the cameras on the head or the body and use depth image as input. Please note vision update should be slow (about 10Hz) and noisy. You should add random blank noise and uniform noise to vision inputs.

4. **Gait control: Walk as human as possible**

   Try any means you can think of to control the robot's gaits. In the original repo some rewards are designed to make good gaits but as you can see, it does not walk in a 'human' way. You should explore features of human gaits and teach the humanoid to imitate.

**References:**
1. Gu, Xinyang, et al. "Advancing humanoid locomotion: Mastering challenging terrains with denoising world model learning." arXiv preprint arXiv:2408.14472 (2024).
2. Peng, Xue Bin, et al. "Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters." ACM Transactions On Graphics (TOG) 41.4 (2022): 1-17.
3. Zhang, Qiang, et al. "Whole-body humanoid robot locomotion with human reference." arXiv preprint arXiv:2402.18294 (2024).
4. Miki, Takahiro, et al. "Learning robust perceptive locomotion for quadrupedal robots in the wild." Science robotics 7.62 (2022): eabk2822. 
5. Agarwal, Ananye, et al. "Legged locomotion in challenging terrains using egocentric vision." Conference on robot learning. PMLR, 2023.

### 2.2 Whole Body Control

Whole body control of full-sized humanoids is a long-standing problem in robotics. The state-of-the-art methods include imitation learning and reinforcement learning with motion capture dataset. You can find motion capture dataset in CMU Mocap Database but you need to retarget to our humanoid in Humanoid-Gym. Here we provide some tasks:

- **Dance**: Pre-define joint trajectories or reaching points to guide the robot to dance.

- **Pick and Place**: Try to pick some things on the ground. The robot needs to bend down or crouch down to pick.

**References:**
1. He, Tairan, et al. "Learning human-to-humanoid real-time whole-body teleoperation." arXiv preprint arXiv:2403.04436 (2024).
2. Cheng, Xuxin, et al. "Expressive whole-body control for humanoid robots." arXiv preprint arXiv:2402.16796 (2024).
3. Seo, Mingyo, et al. "Deep imitation learning for humanoid loco-manipulation through human teleoperation." 2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids). IEEE, 2023.

### 2.3 LLM and Planning

Large language model is powerful for the upper-level planning. You can use LLM to call the low-level controller to complete a long time task.

**References:**
1. Ahn, Michael, et al. "Do as i can, not as i say: Grounding language in robotic affordances." arXiv preprint arXiv:2204.01691 (2022).
2. Huang, Wenlong, et al. "Inner monologue: Embodied reasoning through planning with language models." arXiv preprint arXiv:2207.05608 (2022).

### 2.4 Navigation

Utilizing the Humanoid-Gym codebase, you can create terrain with obstacles and train the robot to navigate. You may use 2-stage training: firstly train to walk, secondly train to navigate. Please explore navigation methods and apply onto the humanoid.

Generally, navigation tasks consist of 2 parts: 
1. Route planning
2. Obstacle avoidance walking 

You can use different algorithms on each part and combine them, or use an end-to-end RL method which maybe hard but more elegant. You can use vision, height map, or obstacle map as observation, but note a high-dim observation may decrease the training speed.

**References:**
1. Zhu, Kai, and Tao Zhang. "Deep reinforcement learning based mobile robot navigation: A review." Tsinghua Science and Technology 26.5 (2021): 674-691.
2. Xiao, Xuesu, et al. "Motion planning and control for mobile robot navigation using machine learning: a survey." Autonomous Robots 46.5 (2022): 569-597.
3. Adamkiewicz, Michal, et al. "Vision-only robot navigation in a neural radiance world." IEEE Robotics and Automation Letters 7.2 (2022): 4606-4613.

### 2.5 Dexterous Hand

The Dynamic Handocer codebase designs a system with two multi-finger hands attached to robot arms and use Multi-Agent RL to conduct throw-and-catch tasks. Please learn to use the codebase and explore related works at first. If you don't have access to real dexterous hands, you can do your whole project in simulation. Here we provide some tasks for you:

- **Generality**: Try to conduct throw-and-catch on various settings

  You can change the distance, directions, or heights of the two hands. To show the generality you may need to test on unseen configs.

- **Robustness**: Try to endure large noise and delay

  You can enlarge the noise and randomization scales and add input delays or action delays which is common in real robot system.

**References:**
1. Huang, Binghao, et al. "Dynamic handover: Throw and catch with bimanual hands." arXiv preprint arXiv:2309.05655 (2023).
2. Zhu, Henry, et al. "Dexterous manipulation with deep reinforcement learning: Efficient, general, and low-cost." 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019.
3. Andrychowicz, OpenAI: Marcin, et al. "Learning dexterous in-hand manipulation." The International Journal of Robotics Research 39.1 (2020): 3-20.

### 2.6 Humanoid-Bench

The Humanoid-Bench codebase provides multiple humanoid tasks and some RL method implementations, including PPO, SAC, DreamerV3, and TD-MPC2. This benchmark has a high-dim action space and complex interactions between robots and the environment so state-of-the-art RL algorithms struggle with most tasks. Here we provide some tasks for you:

- **Hierarchical learning**

  The original work has provided a hierarchical framework, using hand reaching policy as a low-level skill to train on push and package tasks. You can explore new hierarchical frameworks, using more low-level skills on more complex tasks.

- **Teacher-student**

  The observations contain proprioceptive robot state, task-relevant environment obs, egocentric vision, and tactile sensors. You can implement a RMA-like framework, to train a teacher policy using some pivotal inputs and adapt to a student policy using original inputs.

- **Better world model**

  DreamerV3 and TD-MPC2 are model based RL using a world model to predict the dynamics. You can first test whether the baseline models have accurate predictions, and then explore new modules to replace the world model and compare the performance.

**References:**
1. Sferrazza, Carmelo, et al. "Humanoidbench: Simulated humanoid benchmark for whole-body locomotion and manipulation." arXiv preprint arXiv:2403.10506 (2024).
2. Hafner, Danijar, et al. "Mastering diverse domains through world models." arXiv preprint arXiv:2301.04104 (2023).
3. Hansen, Nicklas, Hao Su, and Xiaolong Wang. "Td-mpc2: Scalable, robust world models for continuous control." arXiv preprint arXiv:2310.16828 (2023).

### 2.7 VLA: Vision Language Action

VLA (Vision-Language-Action) is a multimodal artificial intelligence model that integrates visual perception (Vision), language understanding (Language), and action control (Action), enabling robots or intelligent agents to perform physical tasks based on visual and linguistic inputs.

> **Note:** VLA (Vision-Language-Action) models typically demand significant GPU memory (VRAM) due to their multimodal architecture (combining vision, language, and action modules). Before initiating a VLA project, ensure your hardware meets requirements!!!

Here are several related research directions we offer:

- **Non-Transformer Control Policies**

  In the pre-Transformer era, early language-conditioned policies exhibited fundamentally different architectural approaches.

  1. Shridhar, M., Manuelli, L., & Fox, D. (2021). CLIPort: What and Where Pathways for Robotic Manipulation. ArXiv, abs/2109.12098.

  2. Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., & Finn, C. (2022). BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning. ArXiv, abs/2202.02005.

- **Transformer-based Control Policies**

  Since the introduction of the Transformer architecture, control policies have increasingly converged towards similar Transformer-based architectures.

  1. RT-1: Robotics Transformer for Real-World Control at Scale. ArXiv, abs/2212.06817.

  2. Gu, J., Kirmani, S., Wohlhart, P., Lu, Y., Arenas, M.G., Rao, K., Yu, W., Fu, C., Gopalakrishnan, K., Xu, Z., Sundaresan, P., Xu, P., Su, H., Hausman, K., Finn, C., Vuong, Q.H., & Xiao, T. (2023). RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches. ArXiv, abs/2311.01977.

  3. Zhao, T., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. ArXiv, abs/2304.13705.

  4. Li, X., Liu, M., Zhang, H., Yu, C., Xu, J., Wu, H., Cheang, C., Jing, Y., Zhang, W., Liu, H., Li, H., & Kong, T. (2023). Vision-Language Foundation Models as Effective Robot Imitators. ArXiv, abs/2311.01378.

- **Diffusion-based Control Policies**

  Diffusion-Based Action Generation leverages the success of diffusion models in computer vision (CV) to tackle challenges in robotic control and sequential decision-making.

  1. Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. ArXiv, abs/2303.04137.

  2. Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., & Zhu, J. (2024). RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. ArXiv, abs/2410.07864. 
  
  3. Reuss, M., Yagmurlu, Ö.E., Wenzel, F., & Lioutikov, R. (2024). Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals. ArXiv, abs/2407.05996.

- **Large VLA**

  Large Vision-Language-Action (VLA) models are characterized by their extensive multimodal pretraining on diverse datasets encompassing visual, linguistic, and action-oriented data, which facilitates the emergence of advanced affordance understanding capabilities. These models typically employ Transformer-based encoder architectures and are increasingly integrated with large language models (LLMs) to enhance their cross-modal reasoning and task generalization performance.

  1. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. ArXiv, abs/2307.15818.

  2. Belkhale, S., Ding, T., Xiao, T., Sermanet, P., Vuong, Q., Tompson, J., Chebotar, Y., Dwibedi, D., & Sadigh, D. (2024). RT-H: Action Hierarchies Using Language. ArXiv, abs/2403.01823.

  3. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. ArXiv, abs/2310.08864.

  4. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P.R., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., & Finn, C. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. ArXiv, abs/2406.09246.

  5. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L.X., Tanner, J., Vuong, Q., Walling, A., Wang, H., & Zhilinsky, U. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. ArXiv, abs/2410.24164.

- **High-level Task Planners**

  Individual LLMs or multimodal LLMs can generate task plans either through custom-designed frameworks or by fine-tuning on embodied datasets.

  A Task Planner can be implemented as an end-to-end multimodal LLM, leveraging its vast knowledge base for task planning or generating high-level actions, while simultaneously evaluating their feasibility for execution by low-level control policies.

  1. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. Conference on Robot Learning.

  2. PaLM-E: An Embodied Multimodal Language Model. International Conference on Machine Learning.

  Fine-tuning end-to-end models can be highly expensive, leading to the emergence of modular architectures that integrate LLMs and VLMs as task planners, which utilize natural language descriptions as a medium for multimodal information while leveraging LLMs' capability to generate programmatic task plans, with object detection, VLMs, and control policies all invoked through API calls.

  1. Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P.R., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., Sermanet, P., Brown, N., Jackson, T., Luu, L., Levine, S., Hausman, K., & Ichter, B. (2022). Inner Monologue: Embodied Reasoning through Planning with Language Models. ArXiv, abs/2207.05608.

  2. Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., & Garg, A. (2022). ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. 2023 IEEE International Conference on Robotics and Automation (ICRA), 11523-11530.

  3. Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P.R., & Zeng, A. (2022). Code as Policies: Language Model Programs for Embodied Control. 2023 IEEE International Conference on Robotics and Automation (ICRA), 9493-9500.


