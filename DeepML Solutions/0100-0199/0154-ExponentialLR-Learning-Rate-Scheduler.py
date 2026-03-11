class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** epoch)
