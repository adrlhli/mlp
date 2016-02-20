import numpy as np

class MultilayerPerceptron:
    def __init__(self, size, scale=0.1):
        self.w = []
        self.b = []
        for m, n in zip(size[:-1], size[1:]):
            self.w.append(np.random.randn(m, n) * scale)
            self.b.append(np.zeros(n))

    def forward(self, x):
        h = np.array(x, ndmin=2)
        for w, b in zip(self.w[:-1], self.b[:-1]):
            h = np.dot(h, w) + b
            h = h * (h > 0)
        return np.dot(h, self.w[-1]) + self.b[-1]

    def backward(self, x, y, lr=0.1):
        x = np.array(x, ndmin=2)
        y = np.array(y, ndmin=2)
        assert x.shape[0] == y.shape[0]
        batch_size = x.shape[0]
        h = [x]
        for w, b in zip(self.w[:-1], self.b[:-1]):
            h.append(np.dot(h[-1], w) + b)
            h[-1] = h[-1] * (h[-1] > 0)
        h.append(np.dot(h[-1], self.w[-1]) + self.b[-1])
        dh = -(y - h[-1])
        np.copyto(h[-1], 1)
        for w, b, h_i, h_o in zip(self.w[::-1], self.b[::-1], h[-2::-1], h[::-1]):
            dh = dh * np.sign(h_o)
            dw = np.transpose(np.dot(np.transpose(dh), h_i)) / batch_size
            db = np.mean(dh, axis=0)
            dh = np.mean(np.dot(dh, np.transpose(w)), axis=0)
            w -= lr * dw
            b -= lr * db
