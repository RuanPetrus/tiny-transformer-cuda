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
    mx = x.max(-1, keepdims=True)
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
    M = 3

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

def test_layernorm_forward():
    N = 25
    M = 12

    x = np.random.randn(N, M).astype(np.float32)
    w = np.random.randn(M).astype(np.float32)
    b = np.random.randn(M).astype(np.float32)

    eps = 10**(-5)
    var = x.var(-1, keepdims=True)
    mean = x.mean(-1, keepdims=True)

    n = (x - mean) / (var + eps)**(0.5)
    out = n * w + b

    with open(TEMP_PATH + "layernorm_forward.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(M.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        w.tofile(f)
        b.tofile(f)
        out.tofile(f)

def test_encoder_forward():
    B = 10
    T = 7
    A = 6
    C = 20

    x  = np.random.randint(0, A, size=(B, T)).astype(np.int32)
    w  = np.random.randn(A, C).astype(np.float32)
    wp = np.random.randn(T, C).astype(np.float32)
    out = w[x, :] + wp[np.arange(T)]

    with open(TEMP_PATH + "encoder_forward.bin", "wb") as f:
        f.write(B.to_bytes(4, byteorder='little', signed=True))
        f.write(T.to_bytes(4, byteorder='little', signed=True))
        f.write(A.to_bytes(4, byteorder='little', signed=True))
        f.write(C.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        w.tofile(f)
        wp.tofile(f)
        out.tofile(f)

def test_flash_attn_forward():
    B = 5
    N = 1 << 10
    d = 1 << 10

    q  = np.random.randn(B, N, d).astype(np.float32)
    k  = np.random.randn(B, N, d).astype(np.float32)
    v  = np.random.randn(B, N, d).astype(np.float32)

    attn = q@k.transpose(0, 2, 1)
    eattn = softmax(attn)
    out = eattn @ v

    with open(TEMP_PATH + "flash_attn_forward.bin", "wb") as f:
        f.write(B.to_bytes(4, byteorder='little', signed=True))
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        f.write(d.to_bytes(4, byteorder='little', signed=True))
        q.tofile(f)
        k.tofile(f)
        v.tofile(f)
        out.tofile(f)

def main():
    test_matmul()
    test_gelu_forward()
    test_ff_forward()
    test_softmax_forward()
    test_crossentropy_forward()
    test_layernorm_forward()
    test_encoder_forward()
    test_flash_attn_forward()

if __name__ == "__main__":
    main()
