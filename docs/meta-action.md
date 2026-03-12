
Meta Actions. The complete list of these meta actions is provided in Tab. 5. These low-level meta actions are atomic, representing instantaneous kinematic changes in the ego vehicle’s trajectory, and are therefore distinct from high-level driving decisions. A single high-level driving decision within a video segment typically consists of a sequence of such atomic meta actions across both longitudinal and lateral directions. For example, a left lane-change decision may comprise a sequence of steer left, followed by a brief steer right to stabilize the vehicle heading, and then go straight, often accompanied by a gentle accelerate and maintain speed. For each 8-second data sample, we annotate at most one longitudinal and one lateral high-level driving decision, while atomic meta actions are automatically labeled at 10Hz.

Types of atomic meta actions 
Longitudinal
- Gentle accelerate
- Gentle decelerate
- Maintain speed
- Reverse
- Strong accelerate
- Strong decelerate
- Stop


Lateral
- Steer left
- Sharp steer left
- Reverse left
- Go straight
- Steer right
- Sharp steer right
- Reverse right
