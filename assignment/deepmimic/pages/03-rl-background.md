# RL Background - Traditional Humanoid Rewards

Reward terms in traditional RL setups like OpenAI Gym/Gymnasium:

```
total_reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
```

- `healthy_reward`: Every timestep the Humanoid is "alive"
- `forward_reward`: Moving in desired direction 
- `ctrl_cost`: Penalty for large control inputs
- `contact_cost`: Penalty for large contact forces

<br>

<div class="grid grid-cols-2 gap-4">
<div>

**Problems with this approach:**

- Results in unnatural motion
- Requires careful reward tuning
- Adding 10+ reward terms is common
- Still produces robotic/jerky movements

</div>
<div>

<SlidevVideo controls autoplay="once" muted>
  <source src="/videos/unnatural_walking.mp4" type="video/mp4" />
  Your browser does not support videos.
</SlidevVideo>

</div>
</div>