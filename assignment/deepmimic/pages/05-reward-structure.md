# DeepMimic Reward Structure

The reward at each step is a combination of imitation and task rewards:

```
reward = ωI * imitation_reward + ωG * task_reward
```

<br>

<div class="grid grid-cols-2 gap-4">
<div>

**Imitation Reward Components:**
- Pose reward (65%)
- Velocity reward (10%)
- End-effector reward (15%)
- Center-of-mass reward (10%)

</div>
<div>

**Pose Reward:**
- Match joint orientations with reference
- Computed as quaternion difference

**End-effector Reward:**
- Match positions of hands and feet
- Ensures proper placement

</div>
</div>

<br>

This formulation produces natural motions without manually specifying complex biomechanical objectives. 