class EarlyStopping:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.delta_percentage = min_delta
        self.count = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.count = 0
        elif validation_loss > (self.min_validation_loss + self.min_validation_loss*self.delta_percentage):
            self.count += 1
            if self.count >= self.patience:
                return True
        return False