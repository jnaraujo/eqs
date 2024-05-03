package gaussian_elimination

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGE(t *testing.T) {
	m := [][]float64{
		{2, 3, 4, 6},
		{1, 2, 3, 4},
		{3, -4, 0, 10},
	}
	expected := [][]float64{
		{3, -4, 0, 10},
		{0, 5.67, 4, -0.67},
		{0, 0, 0.65, 1.06},
	}
	GaussianElimination(m)
	for i := 0; i < len(m); i++ {
		assert.InDeltaSlice(t, expected[i], m[i], 0.1)
	}
}

func TestBackSubstitution(t *testing.T) {
	actual := [][]float64{
		{3, -4, 0, 10},
		{0, 5.67, 4, -0.67},
		{0, 0, 0.65, 1.06},
	}
	expected := []float64{18.0 / 11.0, -14.0 / 11.0, 18.0 / 11.0}
	solution := BackSubstitution(actual)
	assert.InDeltaSlice(t, expected, solution, 0.1)
}

func TestMaxAbsValColIdx(t *testing.T) {
	A := [][]float64{
		{100, 1, -20.5},
		{3, 20, 20},
		{-1000, 20, -20},
	}
	assert.Equal(t, 2, maxAbsValColIdx(A, 0, 0, 2))
	assert.Equal(t, 1, maxAbsValColIdx(A, 1, 0, 2))
	assert.Equal(t, 0, maxAbsValColIdx(A, 2, 0, 2))
	assert.Equal(t, 1, maxAbsValColIdx(A, 2, 1, 2))
}

func TestMaxAbsValColIdx2(t *testing.T) {
	A := [][]float64{
		{3, -4, 0, 10},
		{0, 5.67, 4, -0.67},
		{0, 0, 0.65, 1.06},
	}
	assert.Equal(t, 1, maxAbsValColIdx(A, 2, 1, 2))
}

func TestRowSwap(t *testing.T) {
	A := [][]float64{
		{100, 1, -20.5},
		{3, 20, 20},
		{-1000, 20, -20},
	}
	swaped := [][]float64{
		{-1000, 20, -20},
		{3, 20, 20},
		{100, 1, -20.5},
	}
	swapRows(A, 0, 2)
	assert.Equal(t, swaped, A)
}
