def linear(learning_rate, min_learning_rate, epoch, learning_rate_decay_epochs):
    """
    Computes the new learning rate
    :param learning_rate: The learning rate
    :param epoch: The current epoch
    :param learning_rate_decay_epochs: Epochs of learning rate decay
    :return: The new learning rate 
    """
    if epoch < learning_rate_decay_epochs:
        rate = epoch/learning_rate_decay_epochs
        return (1 - rate) * learning_rate + rate * min_learning_rate
    return min_learning_rate