import glob
import os
from typing import List, Tuple, Optional, Dict, Union
from secrets import token_hex
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from omegaconf import DictConfig
from flwr.server.history import History
from dataset_prep import CustomSubset
import yaml

# Training Metrics

def comp_accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions wrt the target."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """Return weighted average of accuracy metrics as evaluation."""
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": np.sum(accuracies) / np.sum(examples)}

# Plotters
def get_subset_stats(sub_trainset: CustomSubset):
    """Compute the statistics {label: label's samples counted}  and return them as dict"""
    unq, unq_count = np.unique(sub_trainset.labels, return_counts=True)
    tmp = {int(unq[i]) : int(unq_count[i]) for i in range(len(unq))}
    return tmp

def plot_exp_summary(trainsets: list[CustomSubset], title_str: str, num_classes: int, save_str: str):
    for c_id, sub_trainset in enumerate(trainsets):
        tmp = get_subset_stats(sub_trainset)
        classes = list(tmp.keys())
        counter = list(tmp.values())
        # dy = list(range(len(classes)))
        dy = list(tmp.keys())
        dx = [c_id] * len(classes)
        plt.scatter(dx, dy, s = counter)
    plt.xticks(range(len(trainsets)))
    plt.xlabel('Client')
    plt.yticks(range(num_classes))
    plt.ylabel('Class')
    plt.title(title_str)
    plt.savefig(f"{save_str}.png")
    plt.close()

def plot_client_stats(partitioning: str, id: int, tmp: dict, num_classes: int, save_str_cid, save_str_exp, split: str = ""):
    if split != "":
        f_split = f"{split} split"
    else:
        f_split = ""
    x = list(tmp.keys())
    y = list(tmp.values())
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Class')
    ax1.set_title(f"Client {id} {f_split} with {sum(y)} data samples")
    ax1.set_xticks(x) #label_tags[x], rotation=90)
    # ax1.set_xticklabels(labels=label_tags)
    ax1.set_yticks(y)
    ax1.bar(x,y, width=0.1)
    fig.savefig(f"{save_str_exp}_cid_{id}.png")
    # fig.savefig(f"images/clients_vis/{partitioning}/{id}_cid{save_str_cid}_classes_{num_classes}_summary.png")
    plt.close()

def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : str
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    _, values = zip(*metric_dict["accuracy"])

    # let's extract centralised loss (main metric reported in FedProx paper)
    rounds_loss, values_loss = zip(*hist.losses_centralized)
    # drop zeroth evaluation round before start of training
    # values_loss = values_loss[1:]
    # values = values[1:]
    # rounds_loss = rounds_loss[1:]
    
    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = None,
    default_filename: str = "results.pkl",
    ) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name (so it doesn't.

        overwrite the file).
        """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history}
    if extra_results is not None:
        data = {**data, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_plots(
    local_solvers: List[str], strategy: List[str], var_epochs: bool, momentum_plot=False
):
    """Generate plots for all experiments, saved in directory _static."""
    root_path = "multirun/"
    save_path = "_static/"

    def load_exp(exp_name: str, strat: str, var_epoch: bool):
        exp_dirs = os.path.join(
            root_path,
            f"optimizer_{exp_name.lower()}_strategy_"
            f"{strat.lower()}_var_local_epochs_{var_epoch}",
        )
        # exp_dirs = "outputs"
        exp_files = glob.glob(f"{exp_dirs}/*/*.csv")
        exp_files = glob.glob("outputs/*/*.csv")
        

        exp_df = [pd.read_csv(f) for f in exp_files]
        exp_df = [df for df in exp_df if not df.isna().any().any()]

        assert len(exp_df) >= 1, (
            f"Atleast one results file must contain non-NaN values. "
            f"NaN values found in all seed runs of {exp_df}"
        )
        return exp_df

    def get_confidence_interval(data):
        """Return 95% confidence intervals along with mean."""
        avg = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower = avg - 1.96 * std / np.sqrt(len(data))
        upper = avg + 1.96 * std / np.sqrt(len(data))
        return avg, lower, upper

    # create tuple combination of experiment configuration for plotting
    # [("vanilla", "fedavg", True), ("vanilla", "fednova", True)]
    max_exp_len = max([len(local_solvers), len(strategy)])
    optim_exp_len = int(max_exp_len / len(local_solvers))
    strategy_exp_len = int(max_exp_len / len(strategy))
    var_epochs_len = int(max_exp_len)
    exp_list = list(
        zip(
            local_solvers * optim_exp_len,
            strategy * strategy_exp_len,
            [var_epochs] * var_epochs_len,
        )
    )

    exp_data = [load_exp(*args) for args in exp_list]

    # Iterate over each experiment
    plt.figure()
    title = ""
    for i, data_dfs in enumerate(exp_data):
        # Iterate over multiple seeds of same experiment
        combined_data = np.array([df["test_accuracy"].values for df in data_dfs])

        mean, lower_ci, upper_ci = get_confidence_interval(combined_data)

        epochs = np.arange(1, len(mean) + 1)

        optimizer, server_strategy, variable_epoch = exp_list[i]

        # Assign more readable legends for each plot according to paper
        if optimizer == "proximal" and server_strategy == "FedAvg":
            label = "FedProx"
        elif optimizer.lower() in ["server", "hybrid"]:
            label = optimizer
        elif optimizer.lower() == "vanilla" and momentum_plot:
            label = "No Momentum"
        else:
            label = server_strategy

        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, lower_ci, upper_ci, alpha=0.3)

        if optimizer == "momentum":
            optimizer_label = "SGD-M"
        elif optimizer == "proximal":
            optimizer_label = "SGD w/ Proximal"
        else:
            optimizer_label = "SGD"

        if var_epochs:
            title = f"Local Solver: {optimizer_label}, Epochs ~ U(2, 5)"
        else:
            title = f"Local Solver: {optimizer_label}, Epochs = 2"

        print(
            f"---------------------Local Solver: {optimizer.upper()}, "
            f"Strategy: {server_strategy.upper()} Local Epochs Fixed: {variable_epoch}"
            f"---------------------"
        )
        print(f"Number of valid(not NaN) seeds for this experiment: {len(data_dfs)}")

        print(f"Test Accuracy: {mean[-1]:.2f} ± {upper_ci[-1] - mean[-1]:.2f}")

    if momentum_plot:
        title = "Comparison of Momentum Schemes: FedNova"
        save_name = "momentum_plot"
    else:
        save_name = local_solvers[0]

    plt.ylabel("Test Accuracy %", fontsize=12)
    plt.xlabel("Communication rounds", fontsize=12)
    plt.xlim([0, 103])
    plt.ylim([30, 80])
    plt.legend(loc="lower right", fontsize=12)
    plt.grid()
    plt.title(title, fontsize=15)
    plt.savefig(f"{save_path}testAccuracy_{save_name}_varEpochs_{var_epochs}.png")

    plt.show()

# From visual.ipynb
def plot_progress(eval_file, train_file, cid: int, save_str: str, exp_str: str):
    # Read CSV files
    eval_df = pd.read_csv(eval_file)
    train_df = pd.read_csv(train_file)
    fit_time = 60 * train_df['fit_mins'].mean()
    # Plot accuracies from .csv
    plt.plot(eval_df['server_round'], eval_df['accuracy'], 'o-', markersize=4, label='Eval Accuracy')
    plt.plot(train_df['server_round'], train_df['accuracy'], 'o-', markersize=4, label='Train Accuracy')

    # Add labels and legend
    if cid == 1:
        print(f"{save_str}progress_cid_{cid}.png")
    plt.xlabel('Server round participation')
    plt.ylabel('Accuracy %')
    plt.title(f'Sampled accuracies from {exp_str} client: {cid}, time fit: {fit_time:.3f} s')
    plt.legend(loc = 'lower center')
    plt.savefig(f"{save_str}{exp_str}_progress_cid_{cid}.png")
    # plt.show()
    plt.close()

def plot_clients_progress(exp_file_csv: str):
    """Insert final expirement file of .csv to plot the evaluation and fit accuracy progress of whole party of clients who participated in each round.
    Example: exp_file = 'outputs\\2024-03-30\\23-40-23\\scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=150.csv'
    """
    save_path: str = "./images/comparisons/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    up_dir = os.path.dirname(exp_file_csv)
    config_file = glob.glob(f"{up_dir}/.hydra/config.yaml")[0]
    print(config_file)
    with open(config_file, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    clients = config_data['num_clients']
    exp_name = config_data['exp_name']
    exp_dict = {
        'momentum': 'FedAvg',
        'proximal': 'FedProx',
        'scaffold': 'SCAFFOLD'
    }
    for cid in range(clients):
        # client_cid_cvs = glob.glob(f"{up_dir}/clients/client_*progress_{cid}*")
        cid_eval = glob.glob(f"{up_dir}/clients/client_eval_progress_{cid}*")
        cid_train = glob.glob(f"{up_dir}/clients/client_fit_progress_{cid}*")
        try:
            plot_progress(cid_eval[0], cid_train[0], cid+1, save_str=save_path, exp_str=exp_dict[exp_name])
        except:
            print("Not enough rounds for eval and train accuracy samples")
            FileExistsError()



def compare_alg_on_partitioning(exp_file_csvs: list[str], acc_type: str):
    """
    Plot comparison of an algorithm on different data partitioning modes
    Args:
        exp_file_csvs (str): List of strings pointing to .csv. Must be related to same algorithm and config baseline
        acc_type (str): Chosen accuracy type for data to be compared
    """
    if not os.path.exists('./images/exps/compare_partitioning'):
        os.makedirs('./images/exps/compare_partitioning')  
    exp_dict = {
        'momentum': 'FedAvg',
        'proximal': 'FedProx',
        'scaffold': 'SCAFFOLD'
    }
    accuracy_labels = {
    'train_acc': 'train accuracy',
    'val_acc': 'val accuracy',
    'test_accuracy': 'test accuracy'
    }
    # Load data from CSV into a DataFrame
    dfs = []
    accs = []
    labels: list[str] = []
    len_rounds = 0
    title: str = None
    plt.figure(figsize=(10, 6))
    for i, csv in enumerate(exp_file_csvs):
        up_dir = os.path.dirname(csv)
        config_file = glob.glob(f"{up_dir}/.hydra/config.yaml")[0]
        with open(config_file, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
        df = pd.read_csv(csv)
        dfs.append(df)
        # accs = df[acc_type]
        df_rounds = df['round']
        if i == 0:
            len_rounds = len(df_rounds)
            title = f"{exp_dict[config_data['exp_name']]} data partitioning comparison on {accuracy_labels[acc_type]} w/{config_data['C_fraction']} fraction of #C={config_data['num_clients']} Classes={config_data['num_classes']}"
            exp_str = f"{exp_dict[config_data['exp_name']]}_E={config_data['local_epochs']}_Classes={config_data['num_classes']}"
        if len_rounds > len(df_rounds):
            len_rounds = len(df_rounds)
            # df_rounds = range(len_rounds)
        if config_data['partitioning'] == 'dirichlet':
            partitioning= f"Dir({config_data['alpha']})"
        else:
            partitioning= config_data['partitioning'].upper()
        labels.append(partitioning)
        accs.append(df[acc_type].tolist())
        # print(    labels[-1])
        # plt.plot(df_rounds, accs, label=labels[-1])
    # print(dfs)
    for acc, label in zip(accs, labels):
        plt.plot(range(len_rounds), acc[:len_rounds], label=label)
    # plt.title(f"{exp_dict[exp_name]} {partitioning} R={len_rounds} {config_data['C_fraction']} fraction of #C={config_data['num_clients']} E={config_data['local_epochs']} B={bs}  Classes={config_data['num_classes']}")
    plt.xlabel('Round')
    plt.ylabel('Accuracy %')
    plt.title(title)
    plt.legend(loc='upper left')
    # Show plot
    plt.savefig(f"./images/exps/compare_partitioning/{exp_str}_{acc_type}_summary_.png", bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_summary(exp_file_csv: str):
    """Plot and print summary of a expirement"""
    if not os.path.exists('./images/exps/summaries'):
        os.makedirs('./images/exps/summaries')
    
    up_dir = os.path.dirname(exp_file_csv)
    config_file = glob.glob(f"{up_dir}/.hydra/config.yaml")[0]
    print(exp_file_csv)
    with open(config_file, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    clients = config_data['num_clients']
    exp_name = config_data['exp_name']
    bs= config_data['batch_size']
    exp_dict = {
        'momentum': 'FedAvg',
        'proximal': 'FedProx',
        'scaffold': 'SCAFFOLD'
    }
    exp_str =  os.path.splitext(os.path.basename(exp_file_csv))[0]
    # Load data from CSV into a DataFrame
    df = pd.read_csv(exp_file_csv)
    # Extracting required columns
    try:
        round_values = df['round']
        test_accuracy = df['test_accuracy']
        train_accuracy = df['train_acc']
        val_accuracy = df['val_acc']
        tr,t_r = np.max(train_accuracy), np.argmax(train_accuracy)
        val, v_r = np.max(val_accuracy), np.argmax(val_accuracy)
        test, te_r = np.max(test_accuracy), np.argmax(test_accuracy)
        print(f"Best accuracies on (accuracy,round):  \nTrain-({tr},{t_r}), Val-({val},{v_r}), Test-({test},{te_r})")
        # print(config_data)
        if config_data['partitioning'] == 'dirichlet':
            partitioning= f"Dir({config_data['alpha']})"
        else:
            partitioning= config_data['partitioning'].upper()
        # Plotting the metrics
        plt.figure(figsize=(10, 6))
        plt.plot(round_values, test_accuracy, label='Global model test acc')
        plt.plot(round_values, train_accuracy, label='Weighted train acc')
        plt.plot(round_values, val_accuracy, label='Weighted validation acc')

        # Adding labels and title
        plt.xlabel('Round')
        plt.ylabel('Accuracy %')
        #LR={config_data['optimizer']['lr']}
        plt.title(f"{exp_dict[exp_name]} {partitioning} R={len(round_values)} {config_data['C_fraction']} fraction of #C={config_data['num_clients']} E={config_data['local_epochs']} B={bs}  Classes={config_data['num_classes']}")
        plt.legend(loc='lower center')
        # Show plot
        plt.savefig(f"images/exps/summaries/summary_{exp_str}.png", bbox_inches='tight')
        # plt.show()
        plt.close()
    except:
        print("Corrupted .csv with missing data")




if __name__ == "__main__":
    # for type_epoch_exp in [False, True]:
    #     for solver in ["vanilla", "momentum", "proximal"]:
    #         generate_plots([solver], ["FedAvg", "FedNova"], type_epoch_exp)

    # generate_plots(
    #     ["Hybrid", "Server", "Vanilla"], ["FedNova"], True, momentum_plot=True
    # )
    generate_plots(["momentum"], ["FedAvg"], var_epochs=False)