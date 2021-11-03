from torch import optim
import torch


def train(model, train_loaders, optimizer, criterion, irm, n_epochs, device):
    n_batches = min([len(train_loader) for train_loader in train_loaders])
    for _ in range(n_epochs):
        train_loaders_iter = [iter(train_loader) for train_loader in train_loaders]
        for _ in range(n_batches):
            for train_loader_iter in train_loaders_iter:
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    raise RuntimeError()
                batch = [tensor.to(device) for tensor in batch]
                (X,y) = batch
                scale = torch.tensor(1.).requires_grad_()
                
                optimizer.zero_grad()
                y_pred = model(X)
                loss = torch.zeros(1).to(device)
                # Empirical Risk
                loss += criterion(y_pred*scale, y)
                # Invariance Constraint 
                # TO IMPLEMENT UNBIASED ESTIMATOR
                if irm:
                    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
                    inv_constr = torch.sum(grad ** 2)
                    loss += inv_constr * irm
                loss.backward()
                optimizer.step()
                # TO IMPLEMENT: EVALUATE PERFORMANCES ON VALIDATION

    # TO IMPLEMENT: PICK THE BEST MODEL ON VALIDATION
    # TO IMPLEMENT: SAVE MODEL WITH PROPER NAME
    torch.save(model.state_dict(), "./models/model.pth")