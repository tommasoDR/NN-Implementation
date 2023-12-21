import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(training_loss, test_loss, training_metric, test_metric, loss, metric):
    """
    Plot the results
    """
    path_loss = "../plots/loss.pdf"
    path_metric = "../plots/metric.pdf"

    loss = loss.replace("_", " ")
    metric = metric.replace("_", " ")

    # Create the dataframe of the loss
    training_loss_df = pd.DataFrame(training_loss)
    test_loss_df = pd.DataFrame(test_loss)
    training_loss_df["Set"] = "Training"
    test_loss_df["Set"] = "Test"
    data = pd.concat([training_loss_df, test_loss_df])

    # Plot the loss
    sns.lineplot(data=data, x=data.index, y=0, hue="Set", style="Set", palette=["blue", "red"])

    plt.ylabel(loss, fontsize=15)
    plt.xlabel("Epochs", fontsize=15)
    plt.legend(title='')

    plt.savefig(path_loss, format='pdf', dpi=600, bbox_inches="tight")
    plt.close()

    # Create the dataframe of the metric
    training_metric_df = pd.DataFrame(training_metric)
    test_metric_df = pd.DataFrame(test_metric)
    training_metric_df["Set"] = "Training"
    test_metric_df["Set"] = "Test"
    data = pd.concat([training_metric_df, test_metric_df])

    # Plot the metric
    sns.lineplot(data=data, x=data.index, y=0, hue="Set", style="Set", palette=["blue", "red"])

    plt.ylabel(metric, fontsize=15)
    plt.xlabel("Epochs", fontsize=15)
    plt.legend(title='')

    plt.savefig(path_metric, format='pdf', dpi=600, bbox_inches="tight")
    plt.close()