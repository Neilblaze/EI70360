class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        steps = epoch // self.step_size
        lr = self.initial_lr * (self.gamma ** steps)
        return round(lr, 4)
