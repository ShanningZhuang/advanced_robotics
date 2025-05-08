# DeepMimic Approach

Using motion capture data to shape rewards

<br>

<div class="grid grid-cols-2 gap-4">
<div>

- Reference motion clips provide target poses
- RL objective combines:
  - **Imitation reward**: Match motion capture
  - **Task reward**: Achieve goals (direction, target)

<br>

**Advantages:**

- Natural-looking movements
- Physically plausible responses to perturbations
- Can satisfy task objectives while maintaining style 

</div>
<div>

<img src="/videos/retarget.png" class="w-full" />

</div>
</div>

<div class="absolute bottom-5 right-5 text-xs text-gray-400">
Source: <a href="https://arxiv.org/pdf/2403.04436" target="_blank">Motion Style Transfer (2023)</a>
</div>