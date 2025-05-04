import typing
import random

from src.functions import add, square, sub, vec_dot, vec_add, div
from src.nn.layers import Linear, Sequential
from src.primatives import Vertex, Vector


def parabola(x: Vertex) -> Vertex:
    v1 = add(x, Vertex(4.0))
    v2 = square(v1)
    return v2


def loss_fn(y_pred: Vertex, y: Vertex) -> Vertex:
    return square(sub(y_pred, y))


def opt(params: dict | list | Vertex, nu: float = 0.01) -> None:
    if isinstance(params, dict):
        for p in params.values():
            opt(p, nu)
    elif isinstance(params, list):
        for p in params:
            opt(p, nu)
    elif isinstance(params, Vertex):
        params.value -= nu * params.grad
    else:
        raise ValueError(
            "Parmas must be nested structure (dicts and lists) of Vertex instanceces"
        )


def main():
    random.seed(42)
    m = 5
    n = 1000
    x = [[Vertex(random.uniform(-1, 1)) for _ in range(m)] for _ in range(n)]

    beta = [Vertex(random.gauss(-1, 1)) for _ in range(m)]
    noise = [Vertex(random.gauss(0, 0.1)) for _ in range(n)]
    y = vec_add([vec_dot(beta, x[i]) for i in range(n)], noise)

    model = Sequential([Linear(m, 1)])

    epochs = 100
    for i in range(1, epochs + 1):
        loss_total = 0
        for j in range(n):
            pred_j = model(x[j])[0]
            loss = loss_fn(pred_j, y[j])
            loss = div(loss, Vertex(n))
            loss_total += loss.value

            loss.backward()
            opt(model.parameters, 0.1)
            loss.zero_grad()

        print(f"{i} / {epochs} - {loss_total=}")

    print(f"beta: {beta}")
    print(f"W: {model.parameters['0']['W']}")
    print(f"b: {model.parameters['0']['b']}")


if __name__ == "__main__":
    main()
