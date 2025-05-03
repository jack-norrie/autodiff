from src.primatives import Function, Node


class Add(Function):
    @staticmethod
    def forward(x: Node, y: Node) -> Node:
        z = Node(x.value + y.value, (x, y))
        return z

    @staticmethod
    def backward(*args: tuple[Node, ...]) -> None:
        pass


add = Add()
