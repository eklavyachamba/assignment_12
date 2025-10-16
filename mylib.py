#Name-Eklavya Chauhan
#Roll No- 2311067
#My Library
#This library contains classes for complex numbers, random number generation, Gauss-Jordan elimination, LU decomposition using Crout's method, Cholesky decomposition, Cholesky decomposition, Jacobi method, Gauss-Seidel method, and root-finding methods (Bisection, Regula Falsi, Newton-Raphson, Fixed Point Iteration, NewtonRaphson, Laguerre, MidpointMethod, TrapezoidalMethod).
import math

class MyComplex:
    def __init__(self, real, imag=0.0):
        self.r = real
        self.i = imag

    def display_cmplx(self):
        print(f"{self.r} + {self.i}j")

    def add_cmplx(self, c1, c2):
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i
        return MyComplex(self.r, self.i)

    def sub_cmplx(self, c1, c2):
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i
        return MyComplex(self.r, self.i)

    def mul_cmplx(self, c1, c2):
        self.r = c1.r * c2.r - c1.i * c2.i
        self.i = c1.i * c2.r + c1.r * c2.i
        return MyComplex(self.r, self.i)

    def mod_cmplx(self):
        return math.sqrt(self.r**2 + self.i**2)
def mat_vec_mul(A, x):
    n = len(A)
    m = len(A[0])
    result = [0.0] * n
    for i in range(n):
        for j in range(m):
            result[i] += A[i][j] * x[j]
    return result

def vec_add(x, y):
    return [x[i] + y[i] for i in range(len(x))]

def vec_sub(x, y):
    return [x[i] - y[i] for i in range(len(x))]

def vec_norm(x):
    return sum(xi*xi for xi in x) ** 0.5

def mat_inverse(A):
    n = len(A)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]

    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-14:
            raise ZeroDivisionError("Matrix is singular!")
        for j in range(2*n):
            aug[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]

    return [row[n:] for row in aug]

class random_number:
    def __init__(self, a=1103515245, c=12345, m=32768, seed=1):
        """
        Linear Congruential Generator (LCG):
        X_{n+1} = (a * X_n + c) mod m

        Parameters:
        a, c, m : LCG parameters
        seed : initial seed value
        """
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        """Generate the next random number (integer)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self):
        """Generate the next random number as a float in [0, 1)."""
        return self.next() / self.m

    def generate_floats(self, n):
        """Generate a list of n floats in [0, 1)."""
        floats = []
        for _ in range(n):
            floats.append(self.next_float())
        return floats

class GaussJordan:
    def __init__(self):
        # Read A matrix
        self.A = []
        with open("matrixA.txt", "r") as fA:
            for line in fA:
                if line.strip():
                    self.A.append(list(map(float, line.split())))

        # Read B matrix
        self.B = []
        with open("matrixB.txt", "r") as fB:
            for line in fB:
                if line.strip():
                    self.B.append(float(line.strip()))

        self.n = len(self.A)
        self.output_file = "output1.txt"
        open(self.output_file, "w").close()  # clear file

    def write_output(self, text):
        """Write to both console and file."""
        print(text)
        with open(self.output_file, "a") as f:
            f.write(text + "\n")

    def print_matrix(self, matrix, step=""):
        """Print a matrix with formatting."""
        if step:
            self.write_output(step)
        for row in matrix:
            self.write_output(" ".join(f"{x:10.4f}" for x in row))
        self.write_output("")

    # --------------------------- DETERMINANT ---------------------------
    def determinant(self, M):
        """Compute determinant recursively using Laplace expansion."""
        n = len(M)
        if n == 1:
            return M[0][0]
        if n == 2:
            return M[0][0]*M[1][1] - M[0][1]*M[1][0]
        det = 0
        for c in range(n):
            minor = [row[:c] + row[c+1:] for row in M[1:]]
            det += ((-1)**c) * M[0][c] * self.determinant(minor)
        return det

    # --------------------------- INVERSE ---------------------------
    def inverse(self):
        """Compute inverse using Gauss-Jordan method."""
        n = self.n
        A = [row[:] for row in self.A]
        I = [[float(i == j) for j in range(n)] for i in range(n)]

        for i in range(n):
            # Partial pivoting
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[max_row][i]) < 1e-12:
                self.write_output("Matrix A is singular. Inverse does not exist.")
                return None

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                I[i], I[max_row] = I[max_row], I[i]

            # Normalize pivot row
            pivot = A[i][i]
            for k in range(n):
                A[i][k] /= pivot
                I[i][k] /= pivot

            # Eliminate other rows
            for j in range(n):
                if j != i:
                    factor = A[j][i]
                    for k in range(n):
                        A[j][k] -= factor * A[i][k]
                        I[j][k] -= factor * I[i][k]

        self.print_matrix(I, "Inverse of Matrix A:")
        return I

    # --------------------------- SOLVER ---------------------------
    def solve(self, compute_det=False, compute_inverse=False):
        n = self.n
        if n == 0:
            self.write_output("No equations to solve.")
            return []

        # Compute determinant only if requested
        if compute_det:
            detA = self.determinant(self.A)
            self.write_output(f"Determinant of Matrix A: {detA:.4f}")
            if abs(detA) < 1e-12:
                self.write_output("Matrix A is singular. No unique solution or inverse.")
                return None

        A = [row[:] + [self.B[i]] for i, row in enumerate(self.A)]  # augmented matrix
        self.print_matrix(A, "Initial Augmented Matrix:")

        # Gauss-Jordan elimination
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[max_row][i]) < 1e-12:
                self.write_output("Matrix A is singular. Cannot continue elimination.")
                return None

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                self.print_matrix(A, f"After swapping rows {i+1} and {max_row+1}:")

            pivot = A[i][i]
            for k in range(len(A[i])):
                A[i][k] /= pivot
            self.print_matrix(A, f"After normalizing row {i+1}:")

            for j in range(n):
                if j != i:
                    factor = A[j][i]
                    for k in range(len(A[j])):
                        A[j][k] -= factor * A[i][k]
            self.print_matrix(A, f"After eliminating column {i+1}:")

        # Extract solution
        solution = [A[i][-1] for i in range(n)]
        self.write_output("Final Reduced Row Echelon Form:")
        self.print_matrix(A)
        self.write_output("Solution Vector: " + str(solution))
        self.write_output("Matrix A is non-singular.")

        # Compute inverse only if requested
        if compute_inverse:
            self.inverse()

        return solution
    
import itertools
class LUdecomposition:  # Class for LU decomposition using Crout's method
    def __init__(self, filename):
        self.matrix = self._read_matrix(filename)
        self.output_file = "output2.txt"
        open(self.output_file, "w").close()  # clear previous output

        self.write_output("Input Matrix:")
        self.print_matrix(self.matrix)

        self.L, self.U = self.crout_decomp(self.matrix)
        self.A = self.combine_LU(self.L, self.U)

        # Write results
        self.write_output("Lower Triangular Matrix (L):")
        self.print_matrix(self.L)
        self.write_output("Upper Triangular Matrix (U):")
        self.print_matrix(self.U)
        self.write_output("Storage Matrix (L + U combined):")
        self.print_matrix(self.A)

    def write_output(self, text):
        print(text)
        with open(self.output_file, "a") as f:
            f.write(text + "\n")

    def print_matrix(self, M):
        for row in M:
            formatted_row = " ".join(f"{x:10.4f}" for x in row)
            self.write_output(formatted_row)
        self.write_output("")

    def _read_matrix(self, filename):
        A = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    row = list(map(float, line.split()))
                    A.append(row)
        return A

    # Crout's Method

    def crout_decomp(self, M):
        n = len(M)
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        U = [[0.0 for _ in range(n)] for _ in range(n)]

        # Set diagonal of U to 1
        for i in range(n):
            U[i][i] = 1.0

        for j in range(n):
            # Compute column j of L
            for i in range(j, n):
                sum_l = sum(L[i][k] * U[k][j] for k in range(j))
                L[i][j] = M[i][j] - sum_l

            # Compute row j of U
            for i in range(j + 1, n):
                sum_u = sum(L[j][k] * U[k][i] for k in range(j))
                if L[j][j] == 0:
                    raise ValueError("Zero pivot encountered. Decomposition not possible.")
                U[j][i] = (M[j][i] - sum_u) / L[j][j]

        return L, U

    def combine_LU(self, L, U):
        """Combine L and U into one storage matrix A (as per Crout’s method)."""
        n = len(L)
        A = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i >= j:  # lower triangle (L)
                    A[i][j] = L[i][j]
                else:       # upper triangle (U)
                    A[i][j] = U[i][j]
        return A

    def get_storage_matrix(self):
        """Return the combined LU storage matrix."""
        return self.A
    

class Cholesky:
    # Class for solving linear systems using Cholesky decomposition
    def __init__(self, matrix_file, vector_file):
        # Load coefficient matrix A and RHS vector b from separate files
        self.coeffs = self._read_matrix(matrix_file)
        self.rhs = self._read_rhs(vector_file)
        self.n = len(self.coeffs)

    def _read_matrix(self, filename):
        # Read coefficient matrix A from file
        A = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    row = [float(x) for x in parts]
                    A.append(row)
        return A

    def _read_rhs(self, filename):
        # Read RHS vector b from file
        b = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    b.append(float(parts[0]))  # assuming a single column of values
        return b

    def _cholesky_factor(self):
        # Perform Cholesky decomposition: A = L * L^T
        n = self.n
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                total = 0.0
                for k in range(j):
                    total += L[i][k] * L[j][k]
                if i == j:
                    L[i][j] = math.sqrt(self.coeffs[i][i] - total)
                else:
                    L[i][j] = (self.coeffs[i][j] - total) / L[j][j]
        return L

    def forward_substitution(self, L, b):
        # Solve L y = b
        n = len(b)
        y = [0.0 for _ in range(n)]
        for i in range(n):
            s = sum(L[i][j] * y[j] for j in range(i))  # Sum of products for forward substitution
            y[i] = (b[i] - s) / L[i][i]
        return y

    def backward_substitution(self, U, y):
        # Solve U x = y
        n = len(y)
        x = [0.0 for _ in range(n)]
        for i in reversed(range(n)):
            s = sum(U[i][j] * x[j] for j in range(i + 1, n))  # Sum of products for backward substitution
            x[i] = (y[i] - s) / U[i][i]
        return x

    def solve(self):
        # Main solver function
        L = self._cholesky_factor()
        y = self.forward_substitution(L, self.rhs)
        Lt = [list(row) for row in zip(*L)]  # Transpose of L
        x = self.backward_substitution(Lt, y)
        return x

    def save_solution(self, outfile):
        # Write solution to file
        sol = self.solve()
        with open(outfile, "w") as f:
            f.write("Solution vector x:\n")
            for i, val in enumerate(sol, start=1):
                f.write(f"x{i} = {val:.6f}\n")

class Jacobi:
    def __init__(self, filename, tol=1e-6, max_iter=1000):
        self.A, self.b = self._load_augmented(filename)
        self.n = len(self.A)
        self.tol = tol
        self.max_iter = max_iter
        self.A, self.b = self._make_diagonally_dominant(self.A, self.b)

    def _load_augmented(self, filename):
        A, b = [], []
        with open(filename, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                nums = [float(x) for x in line.split()]
                A.append(nums[:-1])
                b.append(nums[-1])
        return A, b

    def _make_diagonally_dominant(self, A, b):
        n = len(A)
        A_new, b_new = [], []
        used = set()
        for i in range(n):
            best_row, max_val = None, -1
            for r in range(n):
                if r in used:
                    continue
                if abs(A[r][i]) > max_val:
                    max_val = abs(A[r][i])
                    best_row = r
            used.add(best_row)
            A_new.append(A[best_row])
            b_new.append(b[best_row])
        return A_new, b_new

    def step(self, guess):
        nxt = [0.0] * self.n
        for i in range(self.n):
            s = 0.0
            for j in range(self.n):
                if j != i:
                    s += self.A[i][j] * guess[j]
            nxt[i] = (self.b[i] - s) / self.A[i][i]
        return nxt

    def solve(self, f):
        x_curr = [0.0] * self.n
        f.write("Iteration Table (Jacobi Method)\n")
        header = "k\t" + "\t".join([f"x{i+1}" for i in range(self.n)])
        f.write(header + "\n")
        f.write("0\t" + "\t".join(f"{val:.6f}" for val in x_curr) + "\n")

        for k in range(1, self.max_iter + 1):
            x_next = self.step(x_curr)
            f.write(f"{k}\t" + "\t".join(f"{val:.6f}" for val in x_next) + "\n")
            err = max(abs(x_next[i] - x_curr[i]) for i in range(self.n))
            if err < self.tol:
                f.write("\nApproximate solution: " + str([round(val, 6) for val in x_next]) + "\n\n")
                return x_next
            x_curr = x_next

        f.write("\nJacobi method did not converge\n\n")
        return x_curr
 
import itertools

class Jacobi:
    def __init__(self, matrix_filename, vector_filename, tol=1e-6, max_iter=1000):
        # Load matrix A and vector b from their respective files
        self.A = self._load_matrix(matrix_filename)
        self.b = self._load_vector(vector_filename)
        self.n = len(self.A)
        self.tol = tol
        self.max_iter = max_iter
        # Ensure the matrix is diagonally dominant
        self.A, self.b = self._make_diagonally_dominant(self.A, self.b)

    def _load_matrix(self, filename):
        """Load matrix A from a file."""
        A = []
        with open(filename, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                A.append([float(x) for x in line.split()])
        return A

    def _load_vector(self, filename):
        """Load vector b from a file."""
        b = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    b.append(float(line.strip()))
        return b

    def _make_diagonally_dominant(self, A, b):
        """Rearrange rows to make the matrix diagonally dominant, if possible."""
        n = len(A)
        for perm in itertools.permutations(range(n)):
            A_perm = [A[i] for i in perm]
            b_perm = [b[i] for i in perm]
            ok = True
            for i in range(n):
                diag = abs(A_perm[i][i])
                off_diag = sum(abs(A_perm[i][j]) for j in range(n) if j != i)
                if diag < off_diag:
                    ok = False
                    break
            if ok:
                return A_perm, b_perm
        raise RuntimeError("Cannot rearrange into diagonally dominant form")

    def step(self, guess):
        """Perform one Jacobi iteration step."""
        nxt = [0.0] * self.n
        for i in range(self.n):
            s = sum(self.A[i][j] * guess[j] for j in range(self.n) if j != i)
            nxt[i] = (self.b[i] - s) / self.A[i][i]
        return nxt

    def solve(self):
        """Solve the system using the Jacobi method."""
        x_curr = [0.0] * self.n
        for k in range(1, self.max_iter + 1):
            x_next = self.step(x_curr)
            err = max(abs(x_next[i] - x_curr[i]) for i in range(self.n))
            if err < self.tol:
                return x_next
            x_curr = x_next
        raise RuntimeError("Jacobi method did not converge")


class GaussSeidel:
    def __init__(self, filename, tol=1e-6, max_steps=1000):
        A, b = self._load_augmented(filename)
        self.coeffs, self.rhs = self._make_diagonally_dominant(A, b)
        self.n = len(self.coeffs)
        self.tol = tol
        self.max_steps = max_steps

    def _load_augmented(self, filename):
        with open(filename, "r") as f:
            lines = [list(map(float, line.replace("−", "-").split())) for line in f if line.strip()]
        A = [row[:-1] for row in lines]
        b = [row[-1] for row in lines]
        return A, b

    def _make_diagonally_dominant(self, A, b):
        n = len(A)
        for perm in itertools.permutations(range(n)):
            A_perm = [A[i] for i in perm]
            b_perm = [b[i] for i in perm]
            ok = True
            for i in range(n):
                diag = abs(A_perm[i][i])
                off_diag = sum(abs(A_perm[i][j]) for j in range(n) if j != i)
                if diag < off_diag:
                    ok = False
                    break
            if ok:
                return A_perm, b_perm
        raise RuntimeError("Cannot rearrange into diagonally dominant form")

    def solve(self):
        x_curr = [0.0] * self.n
        for step in range(self.max_steps):
            x_old = x_curr.copy()
            for i in range(self.n):
                before = sum(self.coeffs[i][j] * x_curr[j] for j in range(i))
                after = sum(self.coeffs[i][j] * x_old[j] for j in range(i+1, self.n))
                x_curr[i] = (self.rhs[i] - before - after) / self.coeffs[i][i]
            err = max(abs(x_curr[k] - x_old[k]) for k in range(self.n))
            if err < self.tol:
                return x_curr
        raise RuntimeError("Gauss-Seidel did not converge")

class Bisection:
    def __init__(self, func, a, b, tol=1e-6, max_iter=1000, max_expand=50, outfile="output1.txt"):
        self.func = func
        self.left = a
        self.right = b
        self.tol = tol
        self.max_iter = max_iter
        self.max_expand = max_expand
        self.outfile = outfile
        self._bracket_steps = []

    def _find_bracket(self):
        a, b = self.left, self.right
        fa, fb = self.func(a), self.func(b)
        steps = 0

        while fa * fb > 0 and steps < self.max_expand:
            interval = b - a
            a = a - 0.5 * interval
            b = b + 0.5 * interval
            fa, fb = self.func(a), self.func(b)
            steps += 1
            self._bracket_steps.append((steps, a, b, fa, fb))

        if fa * fb > 0:
            raise ValueError("Could not bracket root within expansion limit.")

        self.left, self.right = a, b

    def solve(self):
        self._find_bracket()
        a, b = self.left, self.right
        fa, fb = self.func(a), self.func(b)

        with open(self.outfile, "a") as f:   # append mode
            # Bracketing info
            if self._bracket_steps:
                f.write("Bracketing expansions:\n")
                f.write(f"{'Step':<6}{'a':<15}{'b':<15}{'f(a)':<15}{'f(b)':<15}\n")
                f.write("-" * 70 + "\n")
                for step, a_b, b_b, fa_b, fb_b in self._bracket_steps:
                    f.write(f"{step:<6}{a_b:<15.6f}{b_b:<15.6f}{fa_b:<15.6e}{fb_b:<15.6e}\n")
                f.write("\n")
            f.write(f"Final Bracket Interval: [{self.left:.6f}, {self.right:.6f}]\n\n")

            f.write("==Bisection method==\n\n")
            header = f"{'Iter':<6}{'a':<15}{'f(a)':<15}{'b':<15}{'f(b)':<15}{'midpoint':<15}{'f(midpoint)':<15}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for iteration in range(1, self.max_iter + 1):
                midpoint = 0.5 * (a + b)
                fm = self.func(midpoint)

                f.write(f"{iteration:<6}{a:<15.6f}{fa:<15.6e}{b:<15.6f}{fb:<15.6e}{midpoint:<15.6f}{fm:<15.6e}\n")

                if abs(fm) < self.tol or (b - a) / 2 < self.tol:
                    f.write("\n")
                    f.write(f"Root ≈ {midpoint:.6f} found in {iteration} iterations\n")
                    f.write("="*80 + "\n\n")
                    return midpoint, iteration

                if fa * fm < 0:
                    b, fb = midpoint, fm
                else:
                    a, fa = midpoint, fm

        raise RuntimeError("Bisection did not converge within maximum iterations.")


class RegulaFalsi:
    def __init__(self, func, a, b, tol=1e-6, max_iter=1000, max_expand=50, outfile="output1.txt"):
        self.func = func
        self.left = a
        self.right = b
        self.tol = tol
        self.max_iter = max_iter
        self.max_expand = max_expand
        self.outfile = outfile
        self._bracket_steps = []

    def _find_bracket(self):
        a, b = self.left, self.right
        fa, fb = self.func(a), self.func(b)
        steps = 0

        while fa * fb > 0 and steps < self.max_expand:
            interval = b - a
            a = a - 0.5 * interval
            b = b + 0.5 * interval
            fa, fb = self.func(a), self.func(b)
            steps += 1
            self._bracket_steps.append((steps, a, b, fa, fb))

        if fa * fb > 0:
            raise ValueError("Could not bracket root within expansion limit.")

        self.left, self.right = a, b

    def solve(self):
        self._find_bracket()
        a, b = self.left, self.right
        fa, fb = self.func(a), self.func(b)

        with open(self.outfile, "a") as f:
            if self._bracket_steps:
                f.write("Bracketing expansions:\n")
                f.write(f"{'Step':<6}{'a':<15}{'b':<15}{'f(a)':<15}{'f(b)':<15}\n")
                f.write("-" * 70 + "\n")
                for step, a_b, b_b, fa_b, fb_b in self._bracket_steps:
                    f.write(f"{step:<6}{a_b:<15.6f}{b_b:<15.6f}{fa_b:<15.6e}{fb_b:<15.6e}\n")
                f.write("\n")
            f.write(f"Final Bracket Interval: [{self.left:.6f}, {self.right:.6f}]\n\n")

            f.write("==Regula Falsi method==\n\n")
            header = f"{'Iter':<6}{'a':<15}{'f(a)':<15}{'b':<15}{'f(b)':<15}{'c(false)':<15}{'f(c)':<15}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for iteration in range(1, self.max_iter + 1):
                c = (a * fb - b * fa) / (fb - fa)
                fc = self.func(c)

                f.write(f"{iteration:<6}{a:<15.6f}{fa:<15.6e}{b:<15.6f}{fb:<15.6e}{c:<15.6f}{fc:<15.6e}\n")

                if abs(fc) < self.tol:
                    f.write("\n")
                    f.write(f"Root ≈ {c:.6f} found in {iteration} iterations\n")
                    f.write("="*80 + "\n\n")
                    return c, iteration

                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc

        raise RuntimeError("Regula Falsi did not converge within maximum iterations.")
class NewtonRaphson:
    def __init__(self, func, dfunc, x0=1.0, tol=1e-6, max_iter=1000, outfile="output1.txt"):
        self.func = func # f(x) function
        self.dfunc = dfunc
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.outfile = outfile

    def solve(self):
        x = self.x0
        with open(self.outfile, "a") as f: # append mode
            f.write("==Newton-Raphson method==\n\n")
            header = f"{'Iter':<6}{'x_n':<15}{'f(x_n)':<15}{'f\'(x_n)':<15}{'x_(n+1)':<15}{'|x_(n+1)-x_n|':<15}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for iteration in range(1, self.max_iter + 1): # Iteration loop
                fx = self.func(x)
                dfx = self.dfunc(x)

                if abs(dfx) < 1e-14: #  Prevent division by zero
                    raise ZeroDivisionError("Derivative too small (risk of divergence).")

                x_new = x - fx / dfx # Newton-Raphson formula
                error = abs(x_new - x)

                f.write(f"{iteration:<6}{x:<15.6f}{fx:<15.6e}{dfx:<15.6e}{x_new:<15.6f}{error:<15.6e}\n") # Log iteration details

                if error < self.tol or abs(fx) < self.tol: # Check convergence
                    f.write("\n")
                    f.write(f"Root ≈ {x_new:.6f} found in {iteration} iterations\n")
                    f.write("="*80 + "\n\n")
                    return x_new, iteration

                x = x_new

        raise RuntimeError("Newton-Raphson did not converge within maximum iterations.")
class FixedPoint:
    def __init__(self, gfunc, x0=1.0, tol=1e-6, max_iter=1000, outfile="output1.txt"):
        self.gfunc = gfunc # g(x) function for fixed point iteration
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.outfile = outfile

    def solve(self):
        x = self.x0
        with open(self.outfile, "a") as f: # append mode
            f.write("==Fixed Point Iteration Method==\n\n")
            f.write("Iteration scheme:  x_(n+1) = g(x_n)\n")
            f.write("Stopping criteria: |x_(n+1) - x_n| < tol\n\n")

            header = f"{'Iter':<6}{'x_n':<15}{'g(x_n)':<15}{'|x_(n+1)-x_n|':<15}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for iteration in range(1, self.max_iter + 1): # Iteration loop
                x_new = self.gfunc(x)
                error = abs(x_new - x)

                f.write(f"{iteration:<6}{x:<15.6f}{x_new:<15.6f}{error:<15.6e}\n") # Log iteration details

                if error < self.tol: # Check convergence
                    f.write("\n")
                    f.write(f"Root ≈ {x_new:.6f} found in {iteration} iterations\n")
                    f.write("="*80 + "\n\n")
                    return x_new, iteration

                x = x_new

        raise RuntimeError("Fixed Point Iteration did not converge within maximum iterations.")
    
class NewtonRaphsonSystem:
    def __init__(self, func, x0, tol=1e-6, max_iter=50, outfile="output1.txt"):
        self.func = func              # list of nonlinear equations
        self.x0 = x0[:]               # initial guess
        self.n = len(x0)              # number of unknowns
        self.tol = tol
        self.max_iter = max_iter
        self.outfile = outfile

    # ----- Utility Functions -----
    def _mat_vec_mul(self, A, x):
        n, m = len(A), len(A[0])
        result = [0.0] * n
        for i in range(n):
            for j in range(m):
                result[i] += A[i][j] * x[j]
        return result

    def _vec_add(self, x, y):
        return [x[i] + y[i] for i in range(len(x))]

    def _vec_norm(self, x):
        return sum(xi * xi for xi in x) ** 0.5

    def _mat_inverse(self, A): # inverse by gauss jordan method
        n = len(A)
        aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
        for i in range(n):
            pivot = aug[i][i]
            if abs(pivot) < 1e-14:
                raise ZeroDivisionError("Matrix is singular!")
            for j in range(2 * n):
                aug[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        return [row[n:] for row in aug]

    # ----- Numerical Jacobian -----
    def _jacobian(self, x, h=1e-6):
        n = len(x)
        f0 = self.func(x)
        J = [[0.0] * n for _ in range(n)]
        for j in range(n):
            x_step = x[:]
            x_step[j] += h
            f1 = self.func(x_step)
            for i in range(n):
                J[i][j] = (f1[i] - f0[i]) / h
        return J

    # ----- Main Solver -----
    def solve(self):
        x = self.x0[:]
        with open(self.outfile, "w") as f:
            f.write("== Multivariable Newton-Raphson Method ==\n\n")

            # ---- Dynamic Header ----
            cols = ["Iter"]
            cols += [f"x{i+1}" for i in range(self.n)]
            cols += [f"Δx{i+1}" for i in range(self.n)]
            cols += ["epsilon"]
            header = "".join(f"{c:<12}" for c in cols) + "\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            # ---- Iterations ----
            for iteration in range(1, self.max_iter + 1):
                F = self.func(x)
                J = self._jacobian(x)
                J_inv = self._mat_inverse(J)
                dx = self._mat_vec_mul(J_inv, [-Fi for Fi in F])
                x_new = self._vec_add(x, dx)
                eps = self._vec_norm(dx)

                # Row values
                row = [f"{iteration:<12}"]
                row += [f"{xi:<12.6f}" for xi in x]
                row += [f"{d:<12.6f}" for d in dx]
                row += [f"{eps:<12.6e}"]
                f.write("".join(row) + "\n")

                if eps < self.tol:
                    f.write(f"\nRoot ≈ {x_new} found in {iteration} iterations\n")
                    f.write("=" * 80 + "\n\n")
                    return x_new, iteration

                x = x_new

        raise RuntimeError("Newton-Raphson system did not converge.")


class FixedPointSystem:
    def __init__(self, gfunc, x0, tol=1e-6, max_iter=50, outfile="output1.txt"):
        self.gfunc = gfunc
        self.x0 = x0[:]
        self.n = len(x0)  # number of unknowns
        self.tol = tol
        self.max_iter = max_iter
        self.outfile = outfile

    def _vec_sub(self, x, y):
        return [x[i] - y[i] for i in range(len(x))]

    def _vec_norm(self, x):
        return sum(xi * xi for xi in x) ** 0.5

    def solve(self):
        x = self.x0[:]
        with open(self.outfile, "a") as f:
            f.write("== Multivariable Fixed Point Iteration ==\n\n")

            # ---- Dynamically build header ----
            cols = ["Iter"]
            cols += [f"x{i+1}" for i in range(self.n)]
            cols += [f"x{i+1}'" for i in range(self.n)]
            cols += ["epsilon"]

            header = "".join(f"{c:<12}" for c in cols) + "\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            # ---- Iteration loop ----
            for iteration in range(1, self.max_iter + 1):
                x_new = self.gfunc(x)
                dx = self._vec_sub(x_new, x)
                eps = self._vec_norm(dx)

                # row values
                row = [f"{iteration:<12}"]
                row += [f"{xi:<12.6f}" for xi in x]      # old x
                row += [f"{xi:<12.6f}" for xi in x_new]  # new x
                row += [f"{eps:<12.6e}"]

                f.write("".join(row) + "\n")

                if eps < self.tol:
                    f.write(f"\nRoot ≈ {x_new} found in {iteration} iterations\n")
                    f.write("=" * 80 + "\n\n")
                    return x_new, iteration

                x = x_new

        raise RuntimeError("Fixed Point Iteration system did not converge.")

class LaguerreMethod:
    def __init__(self, coeffs, tol=1e-6, max_iter=100, outfile="output1.txt"):
        # coeffs: list of polynomial coefficients [a_n, a_(n-1), ..., a_0]
        # tol: tolerance for convergence
        # max_iter: maximum iterations
        # outfile: file to write iteration tables and results
        self.coeffs = coeffs[:]  # copy to avoid modifying original
        self.tol = tol
        self.max_iter = max_iter
        self.outfile = outfile

    def poly_eval(self, coeffs, x):
        # Evaluate polynomial and its first two derivatives at x
        n = len(coeffs) - 1
        p = coeffs[0]
        dp = 0
        ddp = 0
        for i in range(1, n + 1):
            ddp = ddp * x + 2 * dp
            dp = dp * x + p
            p = p * x + coeffs[i]
        return p, dp, ddp

    def synthetic_division(self, coeffs, root):
        # Deflate polynomial using synthetic division by (x - root)
        n = len(coeffs) - 1
        new_coeffs = [coeffs[0]]
        for i in range(1, n):
            new_coeffs.append(coeffs[i] + new_coeffs[-1] * root)
        remainder = coeffs[-1] + new_coeffs[-1] * root
        return new_coeffs, remainder

    def find_root(self, coeffs, x0, f):
        # Use Laguerre iteration to find a single real root starting from guess x0
        n = len(coeffs) - 1
        x = x0

        # Write table header (same as slides)
        header = f"{'Iter':<6}{'x_k':<15}{'P(x_k)':<15}{'G(x_k)':<15}{'H(x_k)':<15}{'a_k':<15}{'x_{k+1}':<15}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for iteration in range(1, self.max_iter + 1):
            p, dp, ddp = self.poly_eval(coeffs, x)

            if abs(p) < self.tol:  # Found root
                return x, iteration

            G = dp / p
            H = G * G - ddp / p
            disc = (n - 1) * (n * H - G * G)
            denom1 = G + math.sqrt(abs(disc))
            denom2 = G - math.sqrt(abs(disc))

            if abs(denom1) > abs(denom2):
                a = n / denom1
            else:
                a = n / denom2

            x_new = x - a

            # Write one row of the iteration table
            f.write(f"{iteration:<6}{x:<15.8f}{p:<15.8e}{G:<15.8e}{H:<15.8e}{a:<15.8f}{x_new:<15.8f}\n")

            if abs(x_new - x) < self.tol:  # Convergence check
                return x_new, iteration
            x = x_new

        raise RuntimeError("Laguerre method did not converge.")

    def solve(self, x0=0.0):
        # Find all real roots using Laguerre + deflation
        coeffs = self.coeffs[:]
        roots = []

        with open(self.outfile, "a") as f:
            f.write("== Laguerre Method ==\n\n")

            while len(coeffs) > 2:  # At least quadratic
                f.write(f"--- Finding root for polynomial of degree {len(coeffs)-1} ---\n")
                root, iters = self.find_root(coeffs, x0, f)
                roots.append(round(root, 10))  # round for clean real output

                f.write(f"\nRoot ≈ {root:.10f} found in {iters} iterations\n")
                f.write("="*80 + "\n\n")

                coeffs, remainder = self.synthetic_division(coeffs, root)
                if abs(remainder) > self.tol:
                    f.write(f"Warning: nonzero remainder {remainder}\n")

            # Handle final linear or quadratic
            if len(coeffs) == 2:
                root = -coeffs[1] / coeffs[0]
                roots.append(round(root, 10))
            elif len(coeffs) == 3:
                a, b, c = coeffs
                disc = b*b - 4*a*c
                r1 = (-b + math.sqrt(disc)) / (2*a)
                r2 = (-b - math.sqrt(disc)) / (2*a)
                roots.append(round(r1, 10))
                roots.append(round(r2, 10))
            # Write all roots as a single list at the end
            f.write(f"All roots found: {roots}\n")
            f.write("="*80 + "\n\n")

        return roots
    
class MidpointMethod:
    def __init__(self, f, a, b, N, outfile="output1.txt"):
        self.f = f          # function to integrate
        self.a = a          # lower limit
        self.b = b          # upper limit
        self.N = N          # number of subintervals
        self.outfile = outfile

    def integrate(self):
        h = (self.b - self.a) / self.N

        with open(self.outfile, "a") as f:
            f.write("== Numerical Integration using Midpoint Method ==\n\n")
            f.write(f"Integration range: [{self.a}, {self.b}]\n")
            f.write(f"Number of intervals (N): {self.N}\n")
            f.write(f"Width of each subinterval (h): {h:.6f}\n\n")

            # ---- Header ----
            header = f"{'Index(n)':<12}{'x_n (Midpoint)':<20}{'f(x_n)':<20}{'h*f(x_n)':<20}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            total_area = 0.0

            for n in range(1, self.N + 1):
                # midpoint of each subinterval
                x_left = self.a + (n - 1) * h
                x_right = self.a + n * h
                x_mid = 0.5 * (x_left + x_right)
                fx = self.f(x_mid)
                area = h * fx
                total_area += area

                # write table row
                f.write(f"{n:<12}{x_mid:<20.6f}{fx:<20.6f}{area:<20.6f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Final Approximate Integral (Σ h·f(x_n)) = {total_area:.8f}\n")
            f.write("=" * 70 + "\n\n")

        return total_area

class TrapezoidalMethod:
    def __init__(self, f, a, b, N, outfile="output1.txt"):
        self.f = f          # function to integrate
        self.a = a          # lower limit
        self.b = b          # upper limit
        self.N = N          # number of subintervals
        self.outfile = outfile

    def integrate(self):
        h = (self.b - self.a) / self.N
        x_vals = [self.a + i * h for i in range(self.N + 1)]
        f_vals = [self.f(xi) for xi in x_vals]

        with open(self.outfile, "a") as f:
            f.write("== Numerical Integration using Trapezoidal Method ==\n\n")
            f.write(f"Integration range: [{self.a}, {self.b}]\n")
            f.write(f"Number of intervals (N): {self.N}\n")
            f.write(f"Width of each subinterval (h): {h:.6f}\n\n")

            # ---- Header ----
            header = f"{'Index(n)':<12}{'x_n':<20}{'f(x_n)':<20}{'Weight':<15}{'Weighted f(x_n)':<20}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            total_sum = 0.0
            for n in range(len(x_vals)):
                # weight according to trapezoidal rule
                if n == 0 or n == self.N:
                    w = 1
                else:
                    w = 2
                term = w * f_vals[n]
                total_sum += term

                f.write(f"{n:<12}{x_vals[n]:<20.6f}{f_vals[n]:<20.6f}{w:<15}{term:<20.6f}\n")

            # compute final integral value
            T_N = (h / 2) * total_sum

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Final Approximate Integral (T_N) = {T_N:.8f}\n")
            f.write("=" * 80 + "\n\n")

        return T_N
class Simpsons:
    def __init__(self, f, a, b, N=None, tol=None, outfile="results.txt"):
        self.f = f
        self.a = a
        self.b = b
        self.outfile = outfile

        if tol is not None:
            self.N, _ = self.error_bound_simpson(f, a, b, tol)
        else:
            self.N = N if N is not None else 10

    @staticmethod
    def error_bound_simpson(f, a, b, tol):
        def fourth_derivative(f, x, h=1e-3):
            return (f(x - 2*h) - 4*f(x - h) + 6*f(x) - 4*f(x + h) + f(x + 2*h)) / (h ** 4)

        points = [a + i*(b-a)/200 for i in range(201)]
        f4max = max(abs(fourth_derivative(f, x)) for x in points)
        N = math.ceil((((b - a)**5 * f4max) / (180 * tol)) ** 0.25)
        if N % 2 != 0:
            N += 1
        return N, f4max

    def integrate(self):
        if self.N % 2 != 0:
            self.N += 1

        h = (self.b - self.a) / self.N
        x_vals = [self.a + i * h for i in range(self.N + 1)]
        f_vals = [self.f(xi) for xi in x_vals]

        total_sum = 0.0
        for n in range(len(x_vals)):
            w = 1 if n == 0 or n == self.N else 4 if n % 2 != 0 else 2
            total_sum += w * f_vals[n]

        S_N = (h / 3) * total_sum

        with open(self.outfile, "a") as f:
            f.write("== Simpson's 1/3 Rule Final Result ==\n")
            f.write(f"Integration range: [{self.a}, {self.b}]\n")
            f.write(f"Number of subintervals (N): {self.N}\n")
            f.write(f"Approximate Integral = {S_N:.8f}\n\n")

        return S_N


class MidpointMethodN:
    def __init__(self, f, a, b, N=None, tol=None, outfile="results.txt"):
        self.f = f
        self.a = a
        self.b = b
        self.outfile = outfile

        if tol is not None:
            self.N, _ = self.error_bound_midpoint(f, a, b, tol)
        else:
            self.N = N if N is not None else 10

    @staticmethod
    def error_bound_midpoint(f, a, b, tol):
        def second_derivative(f, x, h=1e-5):
            return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

        points = [a + i*(b-a)/200 for i in range(201)]
        f2max = max(abs(second_derivative(f, x)) for x in points)
        N = math.ceil(math.sqrt(((b - a)**3 * f2max) / (24 * tol)))
        return N, f2max

    def integrate(self):
        h = (self.b - self.a) / self.N
        x_mid = [self.a + (i + 0.5) * h for i in range(self.N)]
        I = h * sum(self.f(xi) for xi in x_mid)

        with open(self.outfile, "a") as f:
            f.write("== Midpoint Method Final Result ==\n")
            f.write(f"Integration range: [{self.a}, {self.b}]\n")
            f.write(f"Number of subintervals (N): {self.N}\n")
            f.write(f"Approximate Integral = {I:.8f}\n\n")

        return I

import matplotlib.pyplot as plt

class MonteCarlo:
    def __init__(self, func, a, b, N=50, rng=None, outfile="results.txt"):
        self.func = func
        self.a = a
        self.b = b
        self.N = N
        self.rng = rng
        self.outfile = outfile

    def integrate(self, rng=None):
        """Perform one Monte Carlo run for current N using provided RNG"""
        rng = rng or self.rng
        xi = [self.a + (self.b - self.a) * rng.next_float() for _ in range(self.N)]
        f_values = [self.func(x) for x in xi]

        mean_f = sum(f_values) / self.N
        mean_f2 = sum(f**2 for f in f_values) / self.N
        sigma_f = math.sqrt(mean_f2 - mean_f**2)
        FN = (self.b - self.a) * mean_f

        return FN, sigma_f

    def convergence(self, start_N=50, step=10, max_N=200, plot=True, save_plots=True):
        """Run Monte Carlo for increasing N with independent samples"""
        Ns, FNs, sigmas = [], [], []

        for N in range(start_N, max_N + 1, step):
            self.N = N
            # Fresh RNG for each N
            rng = type(self.rng)(seed=42 + N) if self.rng else None
            FN, sigma_f = self.integrate(rng=rng)
            Ns.append(N)
            FNs.append(FN)
            sigmas.append(sigma_f)

        # Append convergence data to results.txt
        with open(self.outfile, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write("Monte Carlo Convergence Data\n")
            f.write("="*80 + "\n")
            f.write(f"{'N':<10}{'FN':<20}{'σf':<20}\n")
            f.write("-"*50 + "\n")
            for n, fn, sigma in zip(Ns, FNs, sigmas):
                f.write(f"{n:<10}{fn:<20.8f}{sigma:<20.8f}\n")
            f.write("\n")

        if plot or save_plots:
            # --- Plot FN vs N ---
            plt.figure(figsize=(8,5))
            plt.plot(Ns, FNs, 'o-', label="FN (Integral Estimate)")
            plt.xlabel("Number of Samples (N)")
            plt.ylabel("Integral Estimate (FN)")
            plt.title("Monte Carlo Convergence: FN vs N")
            plt.grid(True)
            plt.legend()
            if save_plots:
                plt.savefig("FN_vs_N.png", dpi=300)
            if plot:
                plt.show()
            plt.close()

            # --- Plot σf vs N ---
            plt.figure(figsize=(8,5))
            plt.plot(Ns, sigmas, 'o-', color='red', label="σf (Standard Deviation)")
            plt.xlabel("Number of Samples (N)")
            plt.ylabel("σf")
            plt.title("Monte Carlo Convergence: σf vs N")
            plt.grid(True)
            plt.legend()
            if save_plots:
                plt.savefig("sigma_vs_N.png", dpi=300)
            if plot:
                plt.show()
            plt.close()

        # Return lists + final FN and σf
        return Ns, FNs, sigmas, FNs[-1], sigmas[-1]
    
class GaussianQuadrature:
    def __init__(self, f, a, b, N, outfile="output1.txt"):
        self.f = f
        self.a = a
        self.b = b
        self.N = N
        self.outfile = outfile

    def _get_legendre_data(self):
        """Return nodes (x_n) and weights (w_n) for Legendre polynomial P_N(x)
           Accurate up to 9 decimal places.
        """
        if self.N == 1:
            x = [0.000000000]
            w = [2.000000000]

        elif self.N == 2:
            x = [-0.577350269, 0.577350269]
            w = [1.000000000, 1.000000000]

        elif self.N == 3:
            x = [-0.774596669, 0.000000000, 0.774596669]
            w = [0.555555556, 0.888888889, 0.555555556]

        elif self.N == 4:
            x = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
            w = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]

        elif self.N == 5:
            x = [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]
            w = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]

        elif self.N == 6:
            x = [-0.9324695142, -0.6612093865, -0.2386191861,
                  0.2386191861, 0.6612093865, 0.9324695142]
            w = [0.1713244924, 0.3607615730, 0.4679139346,
                 0.4679139346, 0.3607615730, 0.1713244924]

        elif self.N == 7:
            x = [-0.9491079123, -0.7415311856, -0.4058451514,
                  0.0000000000, 0.4058451514, 0.7415311856, 0.9491079123]
            w = [0.1294849662, 0.2797053915, 0.3818300505,
                 0.4179591837, 0.3818300505, 0.2797053915, 0.1294849662]

        elif self.N == 8:
            x = [-0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
                  0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565]
            w = [0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
                 0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363]

        elif self.N == 9:
            x = [-0.9681602395, -0.8360311073, -0.6133714327, -0.3242534234, 0.0000000000,
                  0.3242534234, 0.6133714327, 0.8360311073, 0.9681602395]
            w = [0.0812743884, 0.1806481607, 0.2606106964, 0.3123470770, 0.3302393550,
                 0.3123470770, 0.2606106964, 0.1806481607, 0.0812743884]

        elif self.N == 10:
            x = [-0.9739065285, -0.8650633667, -0.6794095683, -0.4333953941, -0.1488743390,
                  0.1488743390, 0.4333953941, 0.6794095683, 0.8650633667, 0.9739065285]
            w = [0.0666713443, 0.1494513492, 0.2190863625, 0.2692667193, 0.2955242247,
                 0.2955242247, 0.2692667193, 0.2190863625, 0.1494513492, 0.0666713443]

        elif self.N == 11:
            x = [-0.9782286581, -0.8870625998, -0.7301520056, -0.5190961292, -0.2695431559,
                  0.0000000000, 0.2695431559, 0.5190961292, 0.7301520056, 0.8870625998, 0.9782286581]
            w = [0.0556685671, 0.1255803695, 0.1862902109, 0.2331937646, 0.2628045445,
                 0.2729250877, 0.2628045445, 0.2331937646, 0.1862902109, 0.1255803695, 0.0556685671]

        elif self.N == 12:
            x = [-0.9815606342, -0.9041172563, -0.7699026742, -0.5873179543,
                 -0.3678314990, -0.1252334085,
                  0.1252334085, 0.3678314990, 0.5873179543, 0.7699026742, 0.9041172563, 0.9815606342]
            w = [0.0471753364, 0.1069393259, 0.1600783285, 0.2031674267, 0.2334925365,
                 0.2491470458, 0.2491470458, 0.2334925365, 0.2031674267, 0.1600783285, 0.1069393259, 0.0471753364]

        elif self.N == 13:
            x = [-0.9841830547, -0.9175983992, -0.8015780907, -0.6423493394, -0.4484927510,
                 -0.2304583159, 0.0000000000, 0.2304583159, 0.4484927510, 0.6423493394, 0.8015780907, 0.9175983992, 0.9841830547]
            w = [0.0404840048, 0.0921214998, 0.1388735102, 0.1781459808, 0.2078160475,
                 0.2262831803, 0.2325515532, 0.2262831803, 0.2078160475, 0.1781459808, 0.1388735102, 0.0921214998, 0.0404840048]

        elif self.N == 14:
            x = [-0.9862838087, -0.9284348837, -0.8272013151, -0.6872929048, -0.5152486363,
                 -0.3191123689, -0.1080549487, 0.1080549487, 0.3191123689, 0.5152486363, 0.6872929048, 0.8272013151, 0.9284348837, 0.9862838087]
            w = [0.0351194603, 0.0801580872, 0.1215185707, 0.1572031672, 0.1855383970,
                 0.2051984637, 0.2152638535, 0.2152638535, 0.2051984637, 0.1855383970, 0.1572031672, 0.1215185707, 0.0801580872, 0.0351194603]

        elif self.N == 15:
            x = [-0.9879925180, -0.9372733924, -0.8482065834, -0.7244177314, -0.5709721726,
                 -0.3941513471, -0.2011940939, 0.0000000000, 0.2011940939, 0.3941513471, 0.5709721726, 0.7244177314, 0.8482065834, 0.9372733924, 0.9879925180]
            w = [0.0307532419, 0.0703660475, 0.1071592205, 0.1395706779, 0.1662692058,
                 0.1861610000, 0.1984314853, 0.2025782419, 0.1984314853, 0.1861610000, 0.1662692058, 0.1395706779, 0.1071592205, 0.0703660475, 0.0307532419]

        elif self.N == 16:
            x = [-0.9894009349, -0.9445750230, -0.8656312024, -0.7554044084, -0.6178762444,
                 -0.4580167776, -0.2816035507, -0.0950125098,
                  0.0950125098, 0.2816035507, 0.4580167776, 0.6178762444, 0.7554044084, 0.8656312024, 0.9445750230, 0.9894009349]
            w = [0.0271524594, 0.0622535239, 0.0951585117, 0.1246289713, 0.1495959888,
                 0.1691565194, 0.1826034150, 0.1894506105, 0.1894506105, 0.1826034150, 0.1691565194, 0.1495959888, 0.1246289713, 0.0951585117, 0.0622535239, 0.0271524594]

        else:
            raise ValueError("Supported N values are 1 to 16 only.")

        return x, w

    def integrate(self):
        x_nodes, w_weights = self._get_legendre_data()
        x_mapped = [((self.b - self.a) / 2) * xi + ((self.b + self.a) / 2) for xi in x_nodes]
        f_values = [self.f(xi) for xi in x_mapped]
        weighted_sum = sum(w_weights[i] * f_values[i] for i in range(self.N))
        I = ((self.b - self.a) / 2) * weighted_sum

        with open(self.outfile, "a") as f:
            f.write("== Numerical Integration using Gaussian Quadrature ==\n\n")
            f.write(f"Integration range: [{self.a}, {self.b}]\n")
            f.write(f"Number of quadrature points (N): {self.N}\n\n")

            header = f"{'Index(n)':<10}{'x_n (Legendre)':<20}{'Mapped x_n':<20}{'w(x_n)':<15}{'f(x_n)':<20}{'w*f(x_n)':<20}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for i in range(self.N):
                term = w_weights[i] * f_values[i]
                f.write(f"{i+1:<10}{x_nodes[i]:<20.9f}{x_mapped[i]:<20.9f}{w_weights[i]:<15.9f}{f_values[i]:<20.9f}{term:<20.9f}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Final Approximate Integral (I) = {I:.9f}\n")
            f.write("=" * 100 + "\n\n")

        return I
