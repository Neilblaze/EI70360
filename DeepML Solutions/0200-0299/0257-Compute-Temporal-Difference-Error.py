import numpy as np

def compute_td_error(v_s: float, reward: float, v_s_prime: float, gamma: float, done: bool) -> float:
    """
    Compute the Temporal Difference (TD) error for a single transition.
    
    Args:
        v_s: Current state value estimate V(s)
        reward: Immediate reward received
        v_s_prime: Next state value estimate V(s')
        gamma: Discount factor (0 <= gamma <= 1)
        done: True if s' is a terminal state
    
    Returns:
        The TD error delta
    """
    td_target = reward + (0 if done else gamma * v_s_prime)
    td_error = td_target - v_s
    return td_error

v_s = 5.0
reward = 1.0
v_s_prime = 10.0
gamma = 0.9
done = False

result = compute_td_error(v_s, reward, v_s_prime, gamma, done)
print(result)