from src.primatives import Vertex
from src.functions import add, mul, square
from src.nn.layers import linear


def parabola(x: Vertex) -> Vertex:
    v1 = add(x, Vertex(4.0))
    v2 = square(v1)
    return v2


def network(W, b, x):
    return linear(W, b, x)[0]


def main():
    x = Vertex(0.0)
    y = network(
        [[Vertex(i) for i in range(10)]],
        [Vertex(i * 10) for i in range(10)],
        [Vertex(i * 100) for i in range(10)],
    )
    print(y)
    # for i in range(20):
    #     print(f"{x=}")
    #     print(f"{y=}")
    #
    #     y.backward()
    #     x.value -= 0.1 * x.grad
    #     y.zero_grad()
    #     print("")


if __name__ == "__main__":
    main()
