import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


def plot_results(training_loss, training_metric, test_loss, test_metric, loss_name, metric_name):
    """
    Plot the results of the training for the loss and the metric
    :param training_loss: The training loss at each epoch
    :param training_metric: The training metric at each epoch
    :param test_loss: The test loss at each epoch
    :param test_metric: The test metric at each epoch
    :param loss_name: The name of the loss function
    :param metric_name: The name of the metric function
    """
    # Create the path for the plots
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)

    path_loss = f"../plots/loss_{current_time}.png"
    path_metric = f"../plots/metric_{current_time}.png"

    loss_name = loss_name.replace("_", " ")
    metric_name = metric_name.replace("_", " ")

    # Create the dataframe of the loss
    training_loss_df = pd.DataFrame(training_loss)
    training_loss_df["Set"] = "Training"

    if test_loss is not None:
        test_loss_df = pd.DataFrame(test_loss)
        test_loss_df["Set"] = "Test"
        data = pd.concat([training_loss_df, test_loss_df])
        pal = ["blue", "red"]
    else:
        pal = ["blue"]
        data = training_loss_df

    # Plot the loss
    sns.lineplot(data=data, x=data.index, y=0, hue="Set", style="Set", palette=pal)

    plt.ylabel(loss_name, fontsize=15)
    plt.ylim(0, 8)
    plt.xlabel("Epochs", fontsize=15)
    plt.legend(title="")

    plt.savefig(path_loss, format="png", dpi=600, bbox_inches="tight")
    plt.close()

    # Create the dataframe of the metric
    training_metric_df = pd.DataFrame(training_metric)
    training_metric_df["Set"] = "Training"

    if test_metric is not None:
        test_metric_df = pd.DataFrame(test_metric)
        test_metric_df["Set"] = "Test"
        data = pd.concat([training_metric_df, test_metric_df])
        pal = ["blue", "red"]
    else:
        pal = ["blue"]
        data = training_metric_df

    # Plot the metric
    sns.lineplot(data=data, x=data.index, y=0, hue="Set", style="Set", palette=pal)

    plt.ylabel(metric_name, fontsize=15)
    plt.ylim(0, 5)
    plt.xlabel("Epochs", fontsize=15)
    plt.legend(title="")

    plt.savefig(path_metric, format="png", dpi=600, bbox_inches="tight")
    plt.close()


def compute_stats(tr_loss, tr_metric, vl_loss, vl_metric, ts_loss=None, ts_metric=None):
    """
    Computes the statistics of the k-fold cross validation
    :param tr_loss: The training loss for each fold
    :param tr_metric: The training metric for each fold
    :param vl_loss: The validation loss for each fold
    :param vl_metric: The validation metric for each fold
    :param ts_loss: The test loss for each fold
    :param ts_metric: The test metric for each fold
    :return: The statistics of the k-fold cross validation
    """
    tr_loss_mean = np.mean(tr_loss)
    tr_loss_std = np.std(tr_loss)

    tr_metric_mean = np.mean(tr_metric)
    tr_metric_std = np.std(tr_metric)

    vl_loss_mean = np.mean(vl_loss)
    vl_loss_std = np.std(vl_loss)

    vl_metric_mean = np.mean(vl_metric)
    vl_metric_std = np.std(vl_metric)

    if ts_loss is None or ts_metric is None:
        stats = {
            "tr_loss_mean": tr_loss_mean,
            "tr_loss_std": tr_loss_std,
            "tr_metric_mean": tr_metric_mean,
            "tr_metric_std": tr_metric_std,
            "vl_loss_mean": vl_loss_mean,
            "vl_loss_std": vl_loss_std,
            "vl_metric_mean": vl_metric_mean,
            "vl_metric_std": vl_metric_std,
        }
        return stats

    ts_loss_mean = np.mean(ts_loss)
    ts_loss_std = np.std(ts_loss)

    ts_metric_mean = np.mean(ts_metric)
    ts_metric_std = np.std(ts_metric)

    stats = {
        "tr_loss_mean": tr_loss_mean,
        "tr_loss_std": tr_loss_std,
        "tr_metric_mean": tr_metric_mean,
        "tr_metric_std": tr_metric_std,
        "vl_loss_mean": vl_loss_mean,
        "vl_loss_std": vl_loss_std,
        "vl_metric_mean": vl_metric_mean,
        "vl_metric_std": vl_metric_std,
        "ts_loss_mean": ts_loss_mean,
        "ts_loss_std": ts_loss_std,
        "ts_metric_mean": ts_metric_mean,
        "ts_metric_std": ts_metric_std,
    }

    return stats
