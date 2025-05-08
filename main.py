import math
import random

import matplotlib.pyplot as plt

from src.nn.Vector import Vector
from src.auto import Vertex
from src.functions import square
from src.nn import Matrix
from src.nn.activation import relu
from src.nn.components import Linear, Sequential
from src.nn.optim import Adam, MomentumSGD


def loss_fn(y_pred: Vertex, y: Vertex) -> Vertex:
    return square((y_pred - y))


def linear_data_gen_experiment():
    random.seed(42)
    m = 5
    n = 10_000
    X = Matrix([[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)])

    beta = Vector([random.gauss(-1, 1) for _ in range(m)])
    noise = Vector([random.gauss(0, 0.1) for _ in range(n)])
    y = X @ beta + noise

    model = Sequential([Linear(m, 1, bias=False)])

    opt = Adam(model.parameters, nu=0.001, beta_1=0.9, beta_2=0.999)

    epochs = 10
    for i in range(1, epochs + 1):
        loss_total = 0
        for j in range(n):
            pred_j = model(X[j])[0]
            loss = loss_fn(pred_j, y[j])
            loss = loss / n
            loss_total += loss.value

            loss.backward()
            opt.step()
            loss.zero_grad()

        print(f"{i} / {epochs} - {loss_total=}")

    print(f"beta: {beta}")
    print(f"W: {model.parameters['0']['W'][0]}")


def non_linear_data_gen_experiment():
    random.seed(42)
    n = 1000
    X = Matrix([[random.uniform(-1, 1)] for i in range(n)])

    def f(X: float) -> float:
        if X < 0:
            return -3 * (X**2) - 2
        else:
            return math.exp(1.5 * X) * math.sin(10 * X)

    y = Vector([f(X[i][0].value) for i in range(n)])
    noise = Vector([random.gauss(0, 0.1) for _ in range(n)])
    y = y + noise

    h = 100
    model = Sequential(
        [
            Linear(1, h, activation=relu, seed=1),
            Linear(h, h, activation=relu, seed=2),
            Linear(h, 1, seed=5),
        ]
    )

    opt = MomentumSGD(model.parameters, nu=0.1, momentum=0.9)

    epochs = 1000
    for i in range(1, epochs + 1):
        loss_total = 0
        for j in range(n):
            pred_j = model(X[j])[0]
            loss = loss_fn(pred_j, y[j])
            loss = loss / n
            loss_total += loss.value

            loss.backward()
            opt.step()
            loss.zero_grad()

        print(f"{i} / {epochs} - {loss_total=}")

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-notebook")

    # Plot raw data points
    xs_data = [X[i][0].value for i in range(n)]
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
    ys_pred = [model(Vector([x]))[0].value for x in xs_curve]
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
    linear_data_gen_experiment()
    # non_linear_data_gen_experiment()


if __name__ == "__main__":
    main()
