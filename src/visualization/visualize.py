import torch
import matplotlib.pyplot as plt

def plot_train_val(m_train, m_val, name_model, period=1, metric='Cross-Entropy Loss', save=True):
    """Plot the evolution of the metric evaluated on the training and validation set during the trainining
    Args:
        m_train: history of the metric evaluated on the training set 
        m_val: history of the metric evaluated on the validation set 
        name_model: name of the experiment
        period: number of epochs between 2 valutation of the metric
        metric: metric used (e.g. Cross-Entropy Loss, Error rate, Accuracy)
        save: equal to True if you want to save the plot
    Returns:
        plot
    """
    plt.figure(figsize=(8,5))
    plt.plot(torch.Tensor(range(1,len(m_train)+1))*period, m_train, 
                color='c', marker='o', ls=':', label=metric+' train')
    plt.plot(torch.Tensor(range(1,len(m_val)+1))*period, m_val, 
                color='m', marker='o', ls=':', label=metric+' val')
    plt.axhline(min(m_val), ls=':',color='black')
    plt.xlabel('Number of Epochs')
    plt.ylabel(metric)
    plt.legend(loc = 'upper right')
    if save==True:
        plt.savefig(f'reports/figures/{name_model}')
    plt.show()