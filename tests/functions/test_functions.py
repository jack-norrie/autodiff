import pytest
from src.functions.functions import add, mult
from src.auto.Vertex import Vertex


class TestAdd:
    def test_add_two_vertices(self):
        x = Vertex(2.0)
        y = Vertex(3.0)
        z = add(x, y)

        assert isinstance(z, Vertex)
        assert z.value == 5.0

        z.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_add_multiple_vertices(self):
        x = Vertex(2.0)
        y = Vertex(3.0)
        w = Vertex(4.0)
        z = add(x, y, w)

        assert isinstance(z, Vertex)
        assert z.value == 9.0

        z.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0
        assert w.grad == 1.0

    def test_add_zero(self):
        x = Vertex(5.0)
        y = Vertex(0.0)
        z = add(x, y)

        assert z.value == 5.0

        z.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_add_negative(self):
        x = Vertex(5.0)
        y = Vertex(-3.0)
        z = add(x, y)

        assert z.value == 2.0

        z.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_add_chain(self):
        x = Vertex(1.0)
        y = Vertex(2.0)
        z = Vertex(3.0)

        w = add(add(x, y), z)

        assert w.value == 6.0

        w.backward()
        assert x.grad == 1.0
        assert y.grad == 1.0
        assert z.grad == 1.0


class TestMult:
    def test_mult_two_vertices(self):
        x = Vertex(2.0)
        y = Vertex(3.0)
        z = mult(x, y)

        assert isinstance(z, Vertex)
        assert z.value == 6.0

        z.backward()
        assert x.grad == 3.0
        assert y.grad == 2.0

    def test_mult_multiple_vertices(self):
        x = Vertex(2.0)
        y = Vertex(3.0)
        w = Vertex(4.0)
        z = mult(x, y, w)

        assert isinstance(z, Vertex)
        assert z.value == 24.0

        z.backward()
        assert x.grad == 12.0
        assert y.grad == 8.0
        assert w.grad == 6.0

    def test_mult_by_zero(self):
        x = Vertex(5.0)
        y = Vertex(0.0)
        z = mult(x, y)

        assert z.value == 0.0

        z.backward()
        assert x.grad == 0.0
        assert y.grad == 5.0

    def test_mult_by_one(self):
        x = Vertex(5.0)
        y = Vertex(1.0)
        z = mult(x, y)

        assert z.value == 5.0

        z.backward()
        assert x.grad == 1.0
        assert y.grad == 5.0

    def test_mult_chain(self):
        x = Vertex(2.0)
        y = Vertex(3.0)
        z = Vertex(4.0)

        w = mult(mult(x, y), z)

        assert w.value == 24.0

        w.backward()
        assert x.grad == 12.0
        assert y.grad == 8.0
        assert z.grad == 6.0


if __name__ == "__main__":
    pytest.main([__file__])
