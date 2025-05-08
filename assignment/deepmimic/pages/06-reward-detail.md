# DeepMimic Reward Structure

The reward function combines imitation of reference motion with task-specific objectives:

<div class="grid grid-cols-2 gap-4">
<div>

## Combined Reward
$r_t = \omega_I r_t^I + \omega_G r_t^G$

Where:
- $r_t^I$ = imitation reward
- $r_t^G$ = task objective reward
- $\omega_I$, $\omega_G$ = respective weights

</div>
<div>

## Imitation Components
$r_t^I = w_p r_t^p + w_v r_t^v + w_e r_t^e + w_c r_t^c$

With fixed weights:
- $w_p = 0.65$ (pose)
- $w_v = 0.1$ (velocity)  
- $w_e = 0.15$ (end-effector)
- $w_c = 0.1$ (center-of-mass)
</div>
</div>
<div class="mt-4">

- **Pose reward**: $r_t^p = \exp [-2 \sum_j ||\hat{q}_t^j \ominus q_t^j||^2]$
- **Velocity reward**: $r_t^v = \exp [-0.1 \sum_j ||\hat{\dot{q}}_t^j - \dot{q}_t^j||^2]$
- **End-effector reward**: $r_t^e = \exp [-40 \sum_e ||\hat{p}_t^e - p_t^e||^2]$
- **Center-of-mass reward**: $r_t^c = \exp [-10 ||\hat{p}_t^c - p_t^c||^2]$

</div>


