import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Residual(nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    first_layers = [nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU()]
    blocks = [ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]
    last_layers = [nn.Linear(hidden_dim, num_classes)]
    return nn.Sequential(*first_layers, *blocks, *last_layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_func = nn.SoftmaxLoss()
    total_loss = 0.0
    total_error = 0
    total_samples = 0
    for X, y in dataloader:
        batch_size = X.shape[0]
        if opt is not None:
            opt.reset_grad()
        logits = model(X)
        loss = loss_func(logits, y)
        predictions = np.argmax(logits.numpy(), axis=1)
        total_error += np.sum(predictions != y.numpy())
        total_loss += float(loss.numpy()) * batch_size
        total_samples += batch_size
        if opt is not None:
            loss.backward()
            opt.step()
    return (float(total_error / total_samples), float(total_loss / total_samples))
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        print(f'Epoch {i + 1}/{epochs} | train loss={train_loss:.4f}, error={train_error:.2%} | test loss={test_loss:.4f}, error={test_error:.2%}', flush=True)
    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
