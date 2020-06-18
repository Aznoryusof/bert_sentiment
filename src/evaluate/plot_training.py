import os, sys
import matplotlib.pyplot as plt
import seaborn as sns

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)


def plot_training(loss_values):

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    plt.savefig(os.path.join(RESULT_DIR, "training_loss.png"))