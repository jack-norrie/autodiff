import typing
import random
import math

from src.functions import add, square, sub, vec_dot, vec_add, div
from src.nn import Linear, Sequential, relu
from src.primatives import Vertex, Vector


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


def linear_data_gen_experiment():
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


def non_linear_data_gen_experiment():
    random.seed(42)
    n = 1000
    x = [[Vertex(random.uniform(-1, 1))] for i in range(n)]

    def f(x: float) -> float:
        return max(-x, math.exp(x) * math.sin(x))

    y = [Vertex(f(x[i][0].value)) for i in range(n)]
    noise = [Vertex(random.gauss(0, 0.1)) for _ in range(n)]
    y = vec_add(y, noise)

    h = 10
    model = Sequential([Linear(1, h, activation=relu), Linear(h, 1)])

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


def main():
    non_linear_data_gen_experiment()


if __name__ == "__main__":
    main()
