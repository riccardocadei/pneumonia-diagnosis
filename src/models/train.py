from torch import optim
import torch

from src.visualization.visualize import  plot_train_val

def train(model, train_loaders, val_loaders, optimizer, criterion, irm, n_epochs, device, plot=True):

    name_model = f'{model.name}_irm_{irm}_ep_{n_epochs}'

    n_batches = min([len(train_loader) for train_loader in train_loaders])
    n_train = sum([len(train_loader.sampler) for train_loader in train_loaders])
    n_val = sum([len(val_loader.sampler) for val_loader in val_loaders])

    train_ers = []
    val_ers = []

    min_val_er =1e10

    for n_epoch in range(n_epochs):
        print(f'Epoch: {n_epoch}/{n_epochs}')
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        train_er = 0
        model.train()
        for _ in range(n_batches):
            loss = torch.zeros(1).to(device)
            for train_loader_iter in train_loaders_iter:
                try:
                    (X,y) = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                X.to(device)
                y.to(device)
                scale = torch.tensor(1.).requires_grad_()
                optimizer.zero_grad()
                y_pred = model(X)

                # Empirical Risk
                bce = criterion(y_pred*scale, y)
                loss += bce
                train_er += bce.item() * X.size(0)
                # Invariance Constraint (TO IMPLEMENT UNBIASED ESTIMATOR)
                if irm:
                    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
                    inv_constr = torch.sum(grad ** 2)
                    loss += inv_constr * irm
            loss.backward()
            optimizer.step()

        train_ers.append(train_er/n_train)
        val_er = validate(model, val_loaders, criterion, device)
        if val_er < min_val_er:
            min_val_er = val_er
            torch.save(model.state_dict(), f"./models/{name_model}.pth")
        val_ers.append(val_er/n_val)

    if plot: plot_train_val(train_ers, val_ers, name_model)

    return 

def validate(model, val_loaders, criterion, device):
    model.eval()
    val_er = 0.0
    for val_loader in val_loaders:
        for (X,y) in val_loader:
            X.to(device)
            y.to(device)
            
            y_pred = model(X)
            loss = criterion(y_pred, y)
            val_er += loss.item() * X.size(0)

    return val_er