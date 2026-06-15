# Task: place the shower maximum at a target depth

We want the longitudinal shower maximum of a 10 GeV electron to sit at a target
layer depth, by tuning the absorber/gap thicknesses (and, if useful, the beam
energy is fixed here). This is a simple, physically-transparent objective for
validating the gradient-grounded design loop: thicker absorber pulls the shower
maximum to earlier layers, and shower max scales roughly as ln(E).

The agent should:
1. Rank which design parameters most strongly and *reliably* move
   `shower_max_depth`.
2. Step along the reliable reverse-mode gradient, clipped to the trust region.
3. Have the Physics-Critic veto any step that relies on an untrusted gradient.
4. Report the final design and whether the predicted change matched the realized
   change.

```yaml
description: >
  Tune absorber thickness (and gap if helpful) so the shower maximum of a 10 GeV
  electron sits near layer 10, using exact reverse-mode gradients with a
  reliability guardrail.
target_observable: shower_max_depth
target_value: 10.0
constraints:
  - "1.0 <= a <= 3.5    # absorber thickness [mm]"
  - "3.0 <= g <= 9.0    # gap thickness [mm]"
  - "energy fixed at 10000 MeV"
max_steps: 6
```
