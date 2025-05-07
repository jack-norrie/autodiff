from src.nn.Vector import Vector
import pytest


class TestConstructor:
    def test_single(self):
        v = Vector(1.0)
        assert v[0].value == 1.0
        assert len(v) == 1

    def test_variadic(self):
        v = Vector(1.0, 2.0, 3.0)
        assert v[0].value == 1.0
        assert v[1].value == 2.0
        assert v[2].value == 3.0
        assert len(v) == 3

    def test_tuple(self):
        v = Vector((1.0, 2.0, 3.0))
        assert v[0].value == 1.0
        assert v[1].value == 2.0
        assert v[2].value == 3.0
        assert len(v) == 3

    def test_list(self):
        v = Vector([1.0, 2.0, 3.0])
        assert v[0].value == 1.0
        assert v[1].value == 2.0
        assert v[2].value == 3.0
        assert len(v) == 3

    def test_no_args(self):
        with pytest.raises(ValueError):
            Vector()


class TestAdd:
    def test_add(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result[0].value == 5.0
        assert result[1].value == 7.0
        assert result[2].value == 9.0
        assert len(result) == 3

    def test_scalar_add(self):
        v = Vector(1.0, 2.0, 3.0)
        result = v + 5.0
        assert result[0].value == 6.0
        assert result[1].value == 7.0
        assert result[2].value == 8.0
        assert len(result) == 3

    def test_radd(self):
        v = Vector(1.0, 2.0, 3.0)
        result = 5.0 + v
        assert result[0].value == 6.0
        assert result[1].value == 7.0
        assert result[2].value == 8.0
        assert len(result) == 3

    def test_different_length_vectors(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0)
        with pytest.raises(ValueError):
            _ = v1 + v2


class TestSub:
    def test_sub(self):
        v1 = Vector(4.0, 5.0, 6.0)
        v2 = Vector(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result[0].value == 3.0
        assert result[1].value == 3.0
        assert result[2].value == 3.0
        assert len(result) == 3

    def test_scalar_sub(self):
        v = Vector(5.0, 7.0, 9.0)
        result = v - 2.0
        assert result[0].value == 3.0
        assert result[1].value == 5.0
        assert result[2].value == 7.0
        assert len(result) == 3

    def test_rsub(self):
        v = Vector(1.0, 2.0, 3.0)
        result = 10.0 - v
        assert result[0].value == 9.0
        assert result[1].value == 8.0
        assert result[2].value == 7.0
        assert len(result) == 3

    def test_different_length_vectors(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0)
        with pytest.raises(ValueError):
            _ = v1 - v2


class TestMul:
    def test_mul(self):
        v1 = Vector(2.0, 3.0, 4.0)
        v2 = Vector(3.0, 2.0, 1.0)
        result = v1 * v2
        assert result[0].value == 6.0
        assert result[1].value == 6.0
        assert result[2].value == 4.0
        assert len(result) == 3

    def test_scalar_mul_right(self):
        v = Vector(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result[0].value == 2.0
        assert result[1].value == 4.0
        assert result[2].value == 6.0
        assert len(result) == 3

    def test_scalar_mul_left(self):
        v = Vector(1.0, 2.0, 3.0)
        result = 2.0 * v
        assert result[0].value == 2.0
        assert result[1].value == 4.0
        assert result[2].value == 6.0
        assert len(result) == 3

    def test_different_length_mul(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0)
        with pytest.raises(ValueError):
            _ = v1 * v2


class TestDiv:
    def test_div(self):
        v1 = Vector(6.0, 8.0, 10.0)
        v2 = Vector(2.0, 4.0, 5.0)
        result = v1 / v2
        assert result[0].value == 3.0
        assert result[1].value == 2.0
        assert result[2].value == 2.0
        assert len(result) == 3

    def test_scalar_div(self):
        v = Vector(2.0, 4.0, 6.0)
        result = v / 2.0
        assert result[0].value == 1.0
        assert result[1].value == 2.0
        assert result[2].value == 3.0
        assert len(result) == 3

    def test_rdiv(self):
        v = Vector(2.0, 4.0, 5.0)
        result = 10.0 / v
        assert result[0].value == 5.0
        assert result[1].value == 2.5
        assert result[2].value == 2.0
        assert len(result) == 3

    def test_different_length_div(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(4.0, 5.0)
        with pytest.raises(ValueError):
            _ = v1 / v2

    def test_div_by_zero(self):
        v1 = Vector(1.0, 2.0, 3.0)
        v2 = Vector(1.0, 0.0, 3.0)
        with pytest.raises(ZeroDivisionError):
            _ = v1 / v2


class TestNeg:
    def test_neg(self):
        v = Vector(1.0, -2.0, 3.0)
        result = -v
        assert result[0].value == -1.0
        assert result[1].value == 2.0
        assert result[2].value == -3.0
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__])
