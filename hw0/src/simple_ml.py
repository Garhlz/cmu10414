import struct
import numpy as np
import gzip

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        raw = f.read()
        # >: Big Endian，大端字节序
        # I: 4 字节无符号整数
        # 从字节序列中解析四个大端序整数
        magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
        assert magic == 2051  # 图像文件

        pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
        X = pixels.reshape(num_images, rows * cols)
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as f:
        raw = f.read()
        magic, num_labels = struct.unpack(">II", raw[:8])
        assert magic == 2049
        assert num_labels == num_images
        labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return X, labels
    ### END YOUR CODE


def softmax_loss(Z, y):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """

    ### BEGIN YOUR CODE
    def softmax(data):
        data -= np.max(data, axis=1, keepdims=True)
        data = np.exp(data)
        return data / np.sum(data, axis=1, keepdims=True)

    P = softmax(Z)
    # 高级索引，使用两个整数数组，同时指定行索引和列索引
    # 得到batch中每个元素的正确标签的概率（softmax之后的）
    correct_probs = P[np.arange(len(y)), y]
    return -np.mean(np.log(correct_probs))
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    total = X.shape[0]
    for start in range(0, total, batch):
        X_batch = X[start : start + batch]
        y_batch = y[start : start + batch]

        # 运行一次forward，得到原始结果
        Z = X_batch @ theta

        # 求出softmax 概率分布
        Z -= np.max(Z, axis=1, keepdims=True)  # 同时减去最大值，防止浮点溢出
        exp_Z = np.exp(Z)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # 求交叉熵的偏导数，dL/dZ = P - Y_onehot，恰好用上高级索引的特性
        # 把正确类别的概率减去 1，其他类别保持不变，也就是 \(P-Y\)
        P[np.arange(len(y_batch)), y_batch] -= 1

        # 求梯度（对batch平均），dL/dtheta = X.T @ dL/dZ / batch_size
        grad = X_batch.T @ P / len(y_batch)

        # 梯度下降
        theta -= grad * lr
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    total = X.shape[0]
    for start in range(0, total, batch):
        X_batch = X[start : start + batch]
        y_batch = y[start : start + batch]
        # 实际batch的长度
        B = len(y_batch)
        Z1 = X_batch @ W1
        H = np.maximum(Z1, 0)  # relu
        Z2 = H @ W2

        Z2 -= np.max(Z2, axis=1, keepdims=True)
        exp_Z2 = np.exp(Z2)
        P = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)  # 交叉熵概率

        # dL / dZ2 = P - Y_onehot
        P[np.arange(B), y_batch] -= 1

        G2 = P / B  # batch内平均
        # 分别对H、W2的偏导
        grad_W2 = H.T @ G2
        G_H = G2 @ W2.T

        G1 = G_H * (Z1 > 0)
        grad_W1 = X_batch.T @ G1

        W2 -= grad_W2 * lr
        W1 -= grad_W1 * lr

    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False):
    """Example function to fully train a softmax regression classifier"""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500, epochs=10, lr=0.5, batch=100):
    """Example function to train two layer neural network"""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    X_te, y_te = parse_mnist(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
