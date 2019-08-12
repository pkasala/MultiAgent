import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):

        self.num_agents = 1
        self.model_name = ""
        self.env= None
        self.warm_up = 0

        self.memory_buffer_size = 100000
        self.memory_batch_size = 512
        self.seed = 2
        self.memory = None

        self.actor_local = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic_local = None
        self.critic_target = None
        self.critic_optimizer = None

        self.noise = True

        self.policy_freg_update = 3
        self.polyak_tau = 0.001
        self.learn_every = 5

        self.discount = 0.991

        self.print_each_steps = 10
        self.store_each_episodes = 10

        self.max_episode = 3000
        self.episode_len = 2000

        print('Using device:', self.DEVICE)

        if self.DEVICE.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')