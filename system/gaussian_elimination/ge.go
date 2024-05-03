package gaussian_elimination

import (
	"math"
)

/*
based on https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
*/
func GaussianElimination(A [][]float64) {
	m := len(A)    // rows
	n := len(A[0]) // cols
	h := 0
	k := 0

	for h < m && k < n {
		iMax := maxAbsValColIdx(A, k, h, m-1)

		if A[iMax][k] == 0 {
			k++
			continue
		}

		swapRows(A, h, iMax)

		for i := h + 1; i < m; i++ {
			f := A[i][k] / A[h][k]
			A[i][k] = 0
			for j := k + 1; j < n; j++ {
				A[i][j] = A[i][j] - A[h][j]*f
			}
		}

		h++
		k++
	}
}

/*
based on https://www.baeldung.com/cs/solving-system-linear-equations back substitution algorithm
*/
func BackSubstitution(A [][]float64) []float64 {
	m := len(A)
	n := len(A[0])
	solution := make([]float64, m)

	for i := m - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < m; j++ {
			sum += solution[j] * A[i][j]
		}
		solution[i] = (1 / A[i][i]) * (A[i][n-1] - sum)
	}

	return solution
}

func swapRows(m [][]float64, a int, b int) {
	m[a], m[b] = m[b], m[a]
}

func maxAbsValColIdx(A [][]float64, k, h, m int) int {
	idx := h
	for i := h; i <= m; i++ {
		if math.Abs(A[i][k]) > math.Abs(A[idx][k]) {
			idx = i
		}
	}
	return idx
}
