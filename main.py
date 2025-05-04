import math
import random

import matplotlib.pyplot as plt

from src.functions import div, square, sub, vec_add, vec_dot
from src.nn import Linear, Sequential, relu
from src.primatives import Vertex


def loss_fn(y_pred: Vertex, y: Vertex) -> Vertex:
    return square(sub(y_pred, y))


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
        if x < 0:
            return -3 * (x**2) - 2
        else:
            return math.exp(1.5 * x) * math.sin(10 * x)

    y = [Vertex(f(x[i][0].value)) for i in range(n)]
    noise = [Vertex(random.gauss(0, 0.1)) for _ in range(n)]
    y = vec_add(y, noise)

    h = 100
    model = Sequential(
        [
            Linear(1, h, activation=relu, seed=1),
            Linear(h, h, activation=relu, seed=2),
            Linear(h, 1, seed=5),
        ]
    )

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

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-notebook")

    # Plot raw data points
    xs_data = [x[i][0].value for i in range(n)]
    ys_data = [y[i].value for i in range(n)]
    plt.scatter(
        xs_data, ys_data, alpha=0.2, color="#1f77b4", label="Training Data", s=10
    )

    # Generate samples from target function
    n_samples = 1000
    xs_curve = [2 * (x / (n_samples - 1)) - 1 for x in range(0, n_samples)]
    ys_true = [f(x) for x in xs_curve]
    plt.plot(
        xs_curve,
        ys_true,
        alpha=1,
        color="#ff7f0e",
        linewidth=2,
        label="True Function",
        ls="--",
    )

    # Plot the model's predictions
    ys_pred = [model([Vertex(x)])[0].value for x in xs_curve]
    plt.plot(
        xs_curve,
        ys_pred,
        alpha=1,
        color="#2ca02c",
        linewidth=2,
        label="Estimated Function",
    )

    plt.title("MLP Function Approximation", fontsize=16)
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output (y)", fontsize=12)

    plt.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    non_linear_data_gen_experiment()


if __name__ == "__main__":
    main()
