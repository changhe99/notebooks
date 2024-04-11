from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
if __name__ == "__main__":
    replay_buffer = ReplayBuffer(capacity=1000)
    replay_buffer.push(1,2,3,4)
    replay_buffer.push(2,3,4,5)
    replay_buffer.push(3,4,5,6)
    
    print(replay_buffer.memory)
    print("***")
    print(replay_buffer.sample(2))
    
    print(len(replay_buffer))