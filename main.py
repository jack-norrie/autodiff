from src.primatives import Node
from src.functions import add, mul
import numpy as np


def main():
    a = Node(np.array(10))
    b = Node(np.array(20))
    c = mul(a, b)

    d = Node(np.array(50))
    e = add(c, d)

    e.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)
    print(e.grad)


if __name__ == "__main__":
    main()
