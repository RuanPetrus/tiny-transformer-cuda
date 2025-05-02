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

def test_gelu_forward():
    N = 40

    x = np.random.randn(N).astype(np.float32)
    GELU_SCALING_FACTOR  = (2.0 / math.pi)**(0.5)

    cube = 0.044715 * x * x * x;
    out = 0.5 * x * (1.0 + np.tanh(GELU_SCALING_FACTOR * (x + cube)))

    with open(TEMP_PATH + "gelu_forward.bin", "wb") as f:
        f.write(N.to_bytes(4, byteorder='little', signed=True))
        x.tofile(f)
        out.tofile(f)

def main():
    test_matmul()
    test_gelu_forward()

if __name__ == "__main__":
    main()
