import numpy as np

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

def main():
    test_matmul()

if __name__ == "__main__":
    test_matmul()
