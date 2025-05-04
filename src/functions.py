from src.primatives import Function, Node


class Add(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value + y.value)
        return z

    @staticmethod
    def backward(x: Node, y: Node) -> tuple[float, ...]:
        return (1.0, 1.0)


add = Add()


class Mul(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value * y.value)
        return z

    @staticmethod
    def backward(x: Node, y: Node) -> tuple[float, ...]:
        return (y.value, x.value)


mul = Mul()


class Square(Function):
    @staticmethod
    def forward(x: Node) -> Node:
        return Node(x.value**2)

    @staticmethod
    def backward(x: Node) -> tuple[float, ...]:
        return (2 * x.value,)


square = Square()
