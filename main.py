from src.primatives import Node
from src.functions import add
import numpy as np


def main():
    a = Node(np.array(10))
    b = Node(np.array(20))

    c = add(a, b)
    print(c)


if __name__ == "__main__":
    main()
