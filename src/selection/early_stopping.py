class EarlyStopping:

    def __init__(self, patience=1, min_delta=0):
        """
        Initialize the early stopping.
        :param patience: number of epochs to wait before stopping the training in case of higher validation loss
        :param min_delta: delta percentage to consider the validation loss higher than the minimum
        """
        self.patience = patience
        self.delta_percentage = min_delta
        self.count = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Check if the training should stop to avoid overfitting.
        :param validation_loss: validation loss on the current epoch
        :return: True if the training should stop, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.count = 0
        elif validation_loss > (self.min_validation_loss + self.min_validation_loss*self.delta_percentage):
            self.count += 1
            if self.count >= self.patience:
                return True
        return False