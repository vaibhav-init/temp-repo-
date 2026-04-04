import re

with open("evaluate_model.py", "r") as f:
    eval_code = f.read()

# 1. Restore the strict stop override (argmax == 0 -> target_speed = 0.0)
target_speed_logic = """                target_speed = sum(uncertainty * self.config.target_speeds)

                if np.argmax(uncertainty) == 0:
                    target_speed = 0.0
"""
# Replace my previous bad block
eval_code = re.sub(
    r"                target_speed = sum\(uncertainty \* self.config.target_speeds\)\s+if target_speed > 0.01:\s+target_speed = max\(target_speed, 3.0\)",
    target_speed_logic,
    eval_code
)

# 2. Add `self.has_started = False` to __init__
init_block_old = r"        self\.stuck_detector = 0\n        self\.force_move = 0"
init_block_new = """        self.stuck_detector = 0
        self.force_move = 0
        self.has_started = False"""
eval_code = re.sub(init_block_old, init_block_new, eval_code)

# 3. Update the stuck detector logic in run_step to support the has_started flag
stuck_old = r"        if speed < 0\.1 and target_speed > 0\.1 and self\.step > grace_period:\n            self\.stuck_detector \+= 1\n        else:\n            self\.stuck_detector = 0"
stuck_new = """        if speed > 0.5:
            self.has_started = True

        # If it hasn't successfully pulled off the line yet, allow stuck_detector to increment
        # so it force-creeps to break the causal trap where stationary=predict_stop.
        wants_to_move = (target_speed > 0.1) or (not self.has_started)
        
        if speed < 0.1 and wants_to_move and self.step > grace_period:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0"""
eval_code = re.sub(stuck_old, stuck_new, eval_code)

with open("evaluate_model.py", "w") as f:
    f.write(eval_code)
