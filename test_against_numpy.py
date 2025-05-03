import numpy as np
import math


TEMP_PATH = "/tmp/tiny_transformer_test_"

def test_matmul():
    N = 20
    K = 10
    M = 15

    x = np.random.randn(N, K).astype(np.float32)
    w = np.random.randn(K, M).astype(np.float32)
    b = np.random.randn(M).astype(np.float32)
    c = x @ w + b

    with open(TEMP_PATH + "matmul.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(K.to_bytes(4, byteorder='little', signed=True))
        f.write(M.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        w.tofile(f)
        b.tofile(f)
        c.tofile(f)

GELU_SCALING_FACTOR  = (2.0 / math.pi)**(0.5)
def gelu(x):
    cube = 0.044715 * x * x * x;
    out = 0.5 * x * (1.0 + np.tanh(GELU_SCALING_FACTOR * (x + cube)))
    return out

def test_gelu_forward():
    N = 40

    x = np.random.randn(N).astype(np.float32)
    out = gelu(x)

    with open(TEMP_PATH + "gelu_forward.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        out.tofile(f)


def test_ff_forward():
    dmodel = 10
    dff = 4 * dmodel
    B = 2
    sequence_len = 3

    x = np.random.randn(B, sequence_len, dmodel).astype(np.float32)
    w0 = np.random.randn(dmodel, dff).astype(np.float32)
    b0 = np.random.randn(dff).astype(np.float32)
    w1 = np.random.randn(dff, dmodel).astype(np.float32)
    b1 = np.random.randn(dmodel).astype(np.float32)

    out0 = x@w0 + b0
    out1 = gelu(out0)
    out = out1 @ w1 + b1

    with open(TEMP_PATH + "ff_forward.bin", "wb") as f:
        f.write(dmodel.to_bytes(4, byteorder='little', signed=True))
        f.write(dff.to_bytes(4, byteorder='little', signed=True))
        f.write(B.to_bytes(4, byteorder='little', signed=True))
        f.write(sequence_len.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        w0.tofile(f)
        b0.tofile(f)
        w1.tofile(f)
        b1.tofile(f)
        out.tofile(f)

def softmax(x):
    mx = x.max()
    x = x - mx
    ex = np.exp(x)
    return ex / ex.sum(-1, keepdims=True)

def test_softmax_forward():
    N = 40
    M = 20

    x = np.random.randn(N, M).astype(np.float32)
    out = softmax(x)

    with open(TEMP_PATH + "softmax_forward.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(M.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        out.tofile(f)

def test_crossentropy_forward():
    N = 2
    M = 2

    x = np.random.randn(N, M).astype(np.float32)
    y = np.random.randint(0, M, size=(N)).astype(np.int32)

    probs = softmax(x)
    yprobs = probs[np.arange(N), y]
    lyprobs = -np.log(yprobs)
    out = lyprobs.mean()

    with open(TEMP_PATH + "crossentropy_forward.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(M.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        y.tofile(f)
        out.tofile(f)

def main():
    test_matmul()
    test_gelu_forward()
    test_ff_forward()
    test_softmax_forward()
    test_crossentropy_forward()

if __name__ == "__main__":
    main()
