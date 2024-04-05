import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_file):
    if not os.path.exists('./images/exps'):
        os.makedirs('./images/exps')
    
    exp_name =  os.path.splitext(os.path.basename(csv_file))[0]
    # Load data from CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Extracting required columns
    round_values = df['round']
    test_accuracy = df['test_accuracy']
    train_accuracy = df['train_acc']
    val_accuracy = df['val_acc']

    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(round_values, test_accuracy, label='Global model test acc')
    plt.plot(round_values, train_accuracy, label='Weighted train acc')
    plt.plot(round_values, val_accuracy, label='Weighted validation acc')

    # Adding labels and title
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(exp_name)
    plt.legend()

    # Show plot
    # plt.show()
    plt.savefig(f"images/exps/{exp_name}.png")
    plt.close()
    

# Example usage:

plot_metrics('multirun/2024-03-14/11-56-42/0/scaffold_scaffold_iid_balanced_Classes=4_Seed=2024_C=10_fraction0.5_B=64_E=1_R=50.csv')
plot_metrics('outputs/2024-03-17/13-30-40/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=16_E=1_R=100.csv')

plot_metrics('outputs/2024-03-17/17-39-36/fedavg_momentum_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=24_E=1_R=100.csv')
plot_metrics('outputs/2024-03-18/11-28-39/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=24_E=1_R=100.csv')
plot_metrics('outputs/2024-03-18/17-49-17/fedavg_proximal_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=24_E=1_R=100.csv')
# scaffold needs changes on params: debug with len()
plot_metrics('outputs/2024-03-19/22-48-12/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=24_E=1_R=25.csv')
plot_metrics('outputs/2024-03-21/11-46-35/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=24_E=1_R=50.csv')
plot_metrics('outputs/2024-03-21/20-38-45/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=100.csv')
plot_metrics('outputs/2024-03-24/21-28-30/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=20.csv')

plot_metrics('outputs/2024-03-25/11-39-51/fedavg_momentum_iid_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=100.csv')
plot_metrics('outputs/2024-03-26/19-59-01/fedavg_proximal_iid_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=100.csv')

# Dc me sum kai / N 
plot_metrics('outputs/2024-03-29/16-32-13/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=20_sum_Dc.csv')
# Dc me aggregate kai * (S / N)
plot_metrics('outputs/2024-03-29/18-27-25/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=20_aggr_Dc.csv')
# Dc me aggregate kai * (S / N) kai eta/g = riza S

# Dc me aggregate 
plot_metrics('outputs/2024-03-29/20-40-06/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=50_aggr_Dc.csv')

plot_metrics('outputs/2024-03-30/10-33-05/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=50_no_mu.csv')

plot_metrics('outputs/2024-03-30/17-05-31/scaffold_scaffold_iid_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=100_amp.csv')
plot_metrics('outputs/2024-03-30/23-40-23/scaffold_scaffold_dirichlet_balanced_Classes=10_Seed=2024_C=10_fraction0.5_B=32_E=1_R=150.csv')