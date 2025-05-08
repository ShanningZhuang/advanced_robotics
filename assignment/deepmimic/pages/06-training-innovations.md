---
layout: two-cols
---
# Key Training Innovations

**Reference State Initialization (RSI):**
- Sample initial states from reference motion
- Enables learning difficult skills like flips
- Character immediately experiences successful states
- Critical for highly dynamic motions

**Early Termination (ET):**
- End episode when character falls
- Prevents wasting computation on failed states
- Focuses learning on successful trajectories
- Improves learning efficiency

::right::

**Why RSI is Important:**
- For backflips, the character must:
  1. Learn proper takeoff
  2. Execute rotation
  3. Land safely
- Without RSI, it may never discover successful states
- With RSI, character experiences all phases from the start

<br>

**Policy Architecture:**
- Neural network with 1024 and 512 hidden units
- State features include character configuration and phase
- Actions specify target angles for PD controllers 