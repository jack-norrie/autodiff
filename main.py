from src.primatives import Node
from src.functions import add, mul, square
import numpy as np


def parabola(x: Node) -> Node:
    v1 = add(x, Node(np.array(4.0)))
    v2 = square(v1)
    return v2


def main():
    x = Node(np.array(0.0))
    for i in range(20):
        y = parabola(x)
        print(f"{x=}")
        print(f"{y=}")

        y.backward()
        x.value -= 0.1 * x.grad
        y.zero_grad()
        print("")


if __name__ == "__main__":
    main()
