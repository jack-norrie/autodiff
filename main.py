from src.primatives import Vertex
from src.functions import add, mul, square, sub
from src.nn.layers import Linear


def parabola(x: Vertex) -> Vertex:
    v1 = add(x, Vertex(4.0))
    v2 = square(v1)
    return v2


def network(W, b, x):
    return linear(W, b, x)[0]


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
    x = [Vertex(0.0) for _ in range(10)]
    y = Vertex(1)

    model = Linear(10, 1)

    for i in range(100):
        y_pred = model(x)[0]
        print(f"{y_pred=}")

        loss = loss_fn(y_pred, y)
        print(f"{loss=}")

        loss.backward()
        opt(model.paramaters, 0.1)

        loss.zero_grad()
        print("")


if __name__ == "__main__":
    main()
