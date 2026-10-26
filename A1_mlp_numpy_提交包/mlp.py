# -*- coding: utf-8 -*-
"""
NumPy 手写全连接神经网络（两层 MLP）图像分类器
核心代码（前馈、反向传播、损失函数、正则化、优化器）全部用 NumPy 实现，不使用 PyTorch/TensorFlow。
"""
import os
import gzip
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# 数据加载：MNIST（IDX 格式，从 torchvision 官方镜像下载）
# ---------------------------------------------------------------------------

# 多个官方镜像，按顺序尝试
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
]

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url, path, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            # 校验 gzip 完整性，防止截断文件
            with gzip.open(path, "rb") as f:
                f.read()
            return
        except Exception as e:
            print(f"  retry {k+1}/{retries} for {os.path.basename(path)}: {e}")
            if os.path.exists(path):
                os.remove(path)
    raise RuntimeError(f"failed to download {url}")


def _load_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2051, f"bad magic {magic}"
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    X = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols)
    return X


def _load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2049, f"bad magic {magic}"
    y = np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)
    return y


def load_mnist(data_dir="./data", val_size=10000, seed=42):
    """加载 MNIST，归一化到 [0,1]，并划分 训练/验证/测试。"""
    os.makedirs(data_dir, exist_ok=True)
    files = {k: os.path.join(data_dir, fname) for k, fname in MNIST_FILES.items()}
    for k, fname in MNIST_FILES.items():
        if not os.path.exists(files[k]):
            print(f"  downloading {fname} ...")
            last_err = None
            for mirror in MNIST_MIRRORS:
                try:
                    _download(f"{mirror}/{fname}", files[k])
                    break
                except RuntimeError as e:
                    last_err = e
            else:
                raise last_err

    Xtr = _load_idx_images(files["train_images"]).astype(np.float32) / 255.0
    ytr = _load_idx_labels(files["train_labels"])
    Xte = _load_idx_images(files["test_images"]).astype(np.float32) / 255.0
    yte = _load_idx_labels(files["test_labels"])

    # 从训练集中划分出验证集（不打乱测试集）
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    Xva, yva = Xtr[:val_size], ytr[:val_size]
    Xtr, ytr = Xtr[val_size:], ytr[val_size:]
    return Xtr, ytr, Xva, yva, Xte, yte


def one_hot(y, num_classes=10):
    t = np.zeros((len(y), num_classes), dtype=np.float32)
    t[np.arange(len(y)), y] = 1.0
    return t


# ---------------------------------------------------------------------------
# 两层全连接网络：前向传播 + 反向传播
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """输入 -> 全连接(ReLU) -> 全连接(线性 logits) -> softmax + 损失"""

    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        # He 初始化（配合 ReLU），输出层用小随机初始化
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes)).astype(np.float32)
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        """前馈传播，缓存中间结果供反向传播使用。"""
        self.X = X
        self.z1 = X @ self.W1 + self.b1        # 第一层线性
        self.h = np.maximum(0.0, self.z1)      # ReLU 激活
        self.z2 = self.h @ self.W2 + self.b2   # 第二层线性（logits）
        return self.z2

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ---- 损失函数（含 L2 正则项），返回 (损失值, 对 logits 的梯度) ----
    def loss_and_dlogits(self, logits, y, loss_name="ce", lam=0.0):
        n = logits.shape[0]
        # 数值稳定的 softmax
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)

        if loss_name == "ce":  # 交叉熵损失
            nll = -np.log(p[np.arange(n), y] + 1e-12)
            data_loss = nll.mean()
            dz = (p - one_hot(y, logits.shape[1])) / n
        elif loss_name == "mse":  # 均方误差损失（对 softmax 输出与 one-hot 标签）
            t = one_hot(y, logits.shape[1])
            data_loss = 0.5 * np.mean(np.sum((p - t) ** 2, axis=1))
            dp = (p - t) / n
            # softmax 的精确链式法则: dz_i = p_i * (dp_i - sum_j p_j*dp_j)
            dz = p * (dp - (dp * p).sum(axis=1, keepdims=True))
        else:
            raise ValueError(f"unknown loss {loss_name}")

        reg_loss = 0.5 * lam * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(data_loss + reg_loss), dz

    def loss(self, X, y, loss_name="ce", lam=0.0):
        logits = self.forward(X)
        return self.loss_and_dlogits(logits, y, loss_name, lam)[0]

    def backward(self, dz):
        """反向传播：由 dL/dz2 逐层求出所有参数梯度。"""
        n = self.X.shape[0]
        lam = self.lam
        # 第二层
        self.dW2 = self.h.T @ dz + lam * self.W2
        self.db2 = dz.sum(axis=0)
        # 第一层
        dh = dz @ self.W2.T
        dz1 = dh * (self.z1 > 0)  # ReLU 导数
        self.dW1 = self.X.T @ dz1 + lam * self.W1
        self.db1 = dz1.sum(axis=0)
        return [self.dW1, self.db1, self.dW2, self.db2]


# ---------------------------------------------------------------------------
# 优化器（全部手写）
# ---------------------------------------------------------------------------

class SGD:
    """随机梯度下降；momentum>0 时为带动量的 SGD（Nesterov 可选）。
    batch_size=len(数据) 即 batch GD，batch_size=1 即 online GD。"""

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            if self.nesterov:
                p += self.momentum * self.v[i] - self.lr * g
            else:
                p += self.v[i]


class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr, self.eps = lr, eps
        self.cache = None

    def step(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.cache[i] += g ** 2
            p -= self.lr * g / (np.sqrt(self.cache[i]) + self.eps)


class RMSProp:
    def __init__(self, lr=0.001, rho=0.99, eps=1e-8):
        self.lr, self.rho, self.eps = lr, rho, eps
        self.cache = None

    def step(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * g ** 2
            p -= self.lr * g / (np.sqrt(self.cache[i]) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Adamax:
    """Adam 的 max 范数变体：用 u_t = max(beta2*u, |g|) 代替二阶矩。"""

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.u = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.u = [np.zeros_like(p) for p in params]
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(g))
            p -= (self.lr / bc1) * self.m[i] / (self.u[i] + self.eps)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": lambda lr=0.01, **kw: SGD(lr=lr, momentum=0.9),
    "nesterov": lambda lr=0.01, **kw: SGD(lr=lr, momentum=0.9, nesterov=True),
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adamax": Adamax,
}


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train(net, optimizer, Xtr, ytr, Xva, yva, epochs=5, batch_size=128,
          loss_name="ce", lam=0.0, seed=0, log_every=1, verbose=True,
          iter_log_every=0):
    """标准 mini-batch 训练循环。返回训练历史。"""
    rng = np.random.default_rng(seed)
    net.lam = lam  # backward 需要正则化强度
    n = len(Xtr)
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "iter": [], "iter_loss": []}
    it = 0
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        Xs, ys = Xtr[idx], ytr[idx]
        for s in range(0, n, batch_size):
            Xb = Xs[s:s + batch_size]
            yb = ys[s:s + batch_size]
            logits = net.forward(Xb)
            loss, dz = net.loss_and_dlogits(logits, yb, loss_name, lam)
            grads = net.backward(dz)
            optimizer.step(net.params(), grads)
            it += 1
            if iter_log_every and it % iter_log_every == 0:
                history["iter"].append(it)
                history["iter_loss"].append(loss)
        tr_loss = net.loss(Xtr[:5000], ytr[:5000], loss_name, lam)
        va_loss = net.loss(Xva, yva, loss_name, lam)
        va_acc = net.accuracy(Xva, yva)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if verbose and epoch % log_every == 0:
            print(f"  epoch {epoch:3d} | train_loss {tr_loss:.4f} | "
                  f"val_loss {va_loss:.4f} | val_acc {va_acc:.4f}")
    net.lam = lam
    return history
