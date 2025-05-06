from src.auto.primitives.Vertex import Vertex
import pytest


class TestAdd:
    @pytest.mark.parametrize(
        ("x", "y", "res"), ((10.0, 5.0, 15.0), (10, 5, 15), (10, -4, 6), (-4, 10, 6))
    )
    def test_forward(self, x, y, res):
        z = Vertex(x) + Vertex(y)
        assert z.value == res

    def test_backward(self):
        x = Vertex(3.0)
        y = Vertex(4.0)
        z = x + y

        z.backward()

        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_in_expression(self):
        x = Vertex(2.0)
        y = Vertex(3.0)

        z1 = (x + y) * x
        assert z1.value == 10.0

        z2 = (x + y) / y
        assert z2.value == 5.0 / 3.0

        z3 = x + y + x
        assert z3.value == 7.0


class TestSub:
    @pytest.mark.parametrize(
        ("x", "y", "res"), ((10.0, 3.0, 7.0), (10, 3, 7), (10, -4, 14), (-4, 10, -14))
    )
    def test_forward(self, x, y, res):
        z = Vertex(x) - Vertex(y)
        assert z.value == res

    def test_backward(self):
        x = Vertex(5.0)
        y = Vertex(2.0)
        z = x - y

        z.backward()

        assert x.grad == 1.0
        assert y.grad == -1.0

    def test_in_expression(self):
        x = Vertex(5.0)
        y = Vertex(2.0)

        z1 = (x - y) * x
        assert z1.value == 15.0

        z2 = (x - y) / y
        assert z2.value == 1.5

        z3 = x - y - x
        assert z3.value == -2.0


class TestMul:
    @pytest.mark.parametrize(
        ("x", "y", "res"),
        ((10.0, 3.0, 30.0), (10, 3, 30), (10, -4, -40), (-4, 10, -40)),
    )
    def test_forward(self, x, y, res):
        z = Vertex(x) * Vertex(y)
        assert z.value == res

    def test_backward(self):
        x = Vertex(3.0)
        y = Vertex(4.0)
        z = x * y

        z.backward()

        assert x.grad == 4.0
        assert y.grad == 3.0

    def test_in_expression(self):
        x = Vertex(2.0)
        y = Vertex(3.0)

        z1 = (x * y) + x
        assert z1.value == 8.0

        z2 = (x * y) / y
        assert z2.value == 2.0

        z3 = x * y * x
        assert z3.value == 12.0


class TestDiv:
    @pytest.mark.parametrize(
        ("x", "y", "res"), ((10.0, 2.0, 5.0), (10, 2, 5), (10, -5, -2), (-10, 5, -2))
    )
    def test_forward(self, x, y, res):
        z = Vertex(x) / Vertex(y)
        assert z.value == res

    def test_backward(self):
        x = Vertex(6.0)
        y = Vertex(2.0)
        z = x / y

        z.backward()

        assert x.grad == 0.5
        assert y.grad == -1.5

    def test_in_expression(self):
        x = Vertex(6.0)
        y = Vertex(2.0)

        z1 = (x / y) + x
        assert z1.value == 9.0

        z2 = (x / y) * y
        assert z2.value == 6.0

        z3 = x / y / x
        assert z3.value == 0.5


class TestNeg:
    @pytest.mark.parametrize(
        "x, expected",
        [
            (5.0, -5.0),
            (-3.0, 3.0),
            (0.0, 0.0),
            (10, -10),
            (-7, 7),
        ],
    )
    def test_forward(self, x, expected):
        v = Vertex(x)
        result = -v
        assert result.value == expected

    def test_backward(self):
        x = Vertex(3.0)
        y = -x

        y.backward()

        assert x.grad == -1.0

    def test_in_expression(self):
        x = Vertex(2.0)
        y = Vertex(3.0)

        z1 = (-x) + y
        assert z1.value == 1.0

        z2 = y - (-x)
        assert z2.value == 5.0

        z3 = (-x) * y
        assert z3.value == -6.0


if __name__ == "__main__":
    pytest.main([__file__])
