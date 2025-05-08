from src.nn.Matrix import Matrix
from src.nn.Vector import Vector
from src.auto import Vertex
import pytest


class TestConstructor:
    def test_basic(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m[0][0].value == 1.0
        assert m[0][1].value == 2.0
        assert m[1][0].value == 3.0
        assert m[1][1].value == 4.0
        assert len(m) == 2
        assert m.shape == (2, 2)

    def test_tuple_indexing(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m[0, 0].value == 1.0
        assert m[0, 1].value == 2.0
        assert m[1, 0].value == 3.0
        assert m[1, 1].value == 4.0

    def test_row_access(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        row = m[0]
        assert isinstance(row, Vector)
        assert row[0].value == 1.0
        assert row[1].value == 2.0
        assert len(row) == 2

    def test_inconsistent_columns(self):
        with pytest.raises(AssertionError):
            Matrix([[1.0, 2.0], [3.0, 4.0, 5.0]])

    def test_empty_matrix(self):
        with pytest.raises(AssertionError):
            Matrix([])


class TestAdd:
    def test_add(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
        result = m1 + m2
        assert result[0, 0].value == 6.0
        assert result[0, 1].value == 8.0
        assert result[1, 0].value == 10.0
        assert result[1, 1].value == 12.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_scalar_add(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = m + 5.0
        assert result[0, 0].value == 6.0
        assert result[0, 1].value == 7.0
        assert result[1, 0].value == 8.0
        assert result[1, 1].value == 9.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_radd(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = 5.0 + m
        assert result[0, 0].value == 6.0
        assert result[0, 1].value == 7.0
        assert result[1, 0].value == 8.0
        assert result[1, 1].value == 9.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_different_shape_matrices(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0]])
        with pytest.raises(ValueError):
            _ = m1 + m2


class TestSub:
    def test_sub(self):
        m1 = Matrix([[5.0, 6.0], [7.0, 8.0]])
        m2 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = m1 - m2
        assert result[0, 0].value == 4.0
        assert result[0, 1].value == 4.0
        assert result[1, 0].value == 4.0
        assert result[1, 1].value == 4.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_scalar_sub(self):
        m = Matrix([[5.0, 7.0], [9.0, 11.0]])
        result = m - 2.0
        assert result[0, 0].value == 3.0
        assert result[0, 1].value == 5.0
        assert result[1, 0].value == 7.0
        assert result[1, 1].value == 9.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_rsub(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = 10.0 - m
        assert result[0, 0].value == 9.0
        assert result[0, 1].value == 8.0
        assert result[1, 0].value == 7.0
        assert result[1, 1].value == 6.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_different_shape_matrices(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0]])
        with pytest.raises(ValueError):
            _ = m1 - m2


class TestMul:
    def test_mul(self):
        m1 = Matrix([[2.0, 3.0], [4.0, 5.0]])
        m2 = Matrix([[3.0, 2.0], [1.0, 4.0]])
        result = m1 * m2
        assert result[0, 0].value == 6.0
        assert result[0, 1].value == 6.0
        assert result[1, 0].value == 4.0
        assert result[1, 1].value == 20.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_scalar_mul_right(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = m * 2.0
        assert result[0, 0].value == 2.0
        assert result[0, 1].value == 4.0
        assert result[1, 0].value == 6.0
        assert result[1, 1].value == 8.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_scalar_mul_left(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        result = 2.0 * m
        assert result[0, 0].value == 2.0
        assert result[0, 1].value == 4.0
        assert result[1, 0].value == 6.0
        assert result[1, 1].value == 8.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_different_shape_mul(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0]])
        with pytest.raises(ValueError):
            _ = m1 * m2


class TestDiv:
    def test_div(self):
        m1 = Matrix([[6.0, 8.0], [10.0, 12.0]])
        m2 = Matrix([[2.0, 4.0], [5.0, 6.0]])
        result = m1 / m2
        assert result[0, 0].value == 3.0
        assert result[0, 1].value == 2.0
        assert result[1, 0].value == 2.0
        assert result[1, 1].value == 2.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_scalar_div(self):
        m = Matrix([[2.0, 4.0], [6.0, 8.0]])
        result = m / 2.0
        assert result[0, 0].value == 1.0
        assert result[0, 1].value == 2.0
        assert result[1, 0].value == 3.0
        assert result[1, 1].value == 4.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_rdiv(self):
        m = Matrix([[2.0, 4.0], [5.0, 10.0]])
        result = 10.0 / m
        assert result[0, 0].value == 5.0
        assert result[0, 1].value == 2.5
        assert result[1, 0].value == 2.0
        assert result[1, 1].value == 1.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_different_shape_div(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0]])
        with pytest.raises(ValueError):
            _ = m1 / m2

    def test_div_by_zero(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[1.0, 0.0], [3.0, 4.0]])
        with pytest.raises(ZeroDivisionError):
            _ = m1 / m2


class TestNeg:
    def test_neg(self):
        m = Matrix([[1.0, -2.0], [3.0, -4.0]])
        result = -m
        assert result[0, 0].value == -1.0
        assert result[0, 1].value == 2.0
        assert result[1, 0].value == -3.0
        assert result[1, 1].value == 4.0
        assert len(result) == 2
        assert result.shape == (2, 2)


class TestMatMul:
    def test_matrix_matrix_mul(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
        result = m1 @ m2

        assert result[0, 0].value == 19.0
        assert result[0, 1].value == 22.0
        assert result[1, 0].value == 43.0
        assert result[1, 1].value == 50.0
        assert len(result) == 2
        assert result.shape == (2, 2)

    def test_matrix_vector_mul(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        v = Vector(5.0, 6.0)
        result = m @ v

        assert result[0, 0].value == 17.0
        assert result[1, 0].value == 39.0
        assert len(result) == 2
        assert result.shape == (2, 1)

    def test_non_square_matrix_mul(self):
        m1 = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m2 = Matrix([[7.0], [8.0], [9.0]])
        result = m1 @ m2

        assert result[0, 0].value == 50.0
        assert result[1, 0].value == 122.0
        assert len(result) == 2
        assert result.shape == (2, 1)

    def test_incompatible_dimensions(self):
        m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = Matrix([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

        with pytest.raises(AssertionError):
            _ = m2 @ m1

    def test_vector_with_wrong_dimension(self):
        m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        v = Vector(5.0, 6.0, 7.0)

        with pytest.raises(AssertionError):
            _ = m @ v


if __name__ == "__main__":
    pytest.main([__file__])
