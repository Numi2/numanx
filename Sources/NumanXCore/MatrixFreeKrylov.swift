import Foundation

public protocol NXLinearOperator: Sendable {
    var dimension: Int { get }
    func apply(_ x: [Float], into y: inout [Float]) throws
}

public protocol NXPreconditioner: Sendable {
    var dimension: Int { get }
    func apply(_ residual: [Float], iteration: Int, into correction: inout [Float]) throws
}

public struct NXIdentityPreconditioner: NXPreconditioner {
    public let dimension: Int
    public init(dimension: Int) { self.dimension = dimension }
    public func apply(_ residual: [Float], iteration: Int, into correction: inout [Float]) throws {
        guard residual.count == dimension else { throw NXError.invalidState("preconditioner dimension") }
        correction = residual
    }
}

public struct NXKrylovResult: Equatable, Sendable {
    public var solution: [Float]
    public var iterations: Int
    public var residualNorm: Float
    public var converged: Bool
}

public enum NXVectorMath {
    @inline(__always) public static func dot(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else { throw NXError.invalidState("dot dimension") }
        var sum: Double = 0
        var compensation: Double = 0
        for i in a.indices {
            let product = Double(a[i]) * Double(b[i])
            let corrected = product - compensation
            let next = sum + corrected
            compensation = (next - sum) - corrected
            sum = next
        }
        let value = Float(sum)
        guard value.isFinite else { throw NXError.numericalBreakdown("dot overflow") }
        return value
    }

    @inline(__always) public static func norm(_ x: [Float]) throws -> Float {
        let squared = try dot(x, x)
        guard squared >= 0 else { throw NXError.numericalBreakdown("negative norm") }
        return sqrt(squared)
    }

    public static func axpy(alpha: Float, x: [Float], y: inout [Float]) throws {
        guard alpha.isFinite, x.count == y.count else { throw NXError.invalidState("axpy") }
        for i in y.indices {
            let value = y[i] + alpha * x[i]
            guard value.isFinite else { throw NXError.numericalBreakdown("axpy nonfinite") }
            y[i] = value
        }
    }

    public static func scaled(_ x: [Float], by alpha: Float) throws -> [Float] {
        guard alpha.isFinite else { throw NXError.invalidState("scale") }
        return try x.map {
            let value = $0 * alpha
            guard value.isFinite else { throw NXError.numericalBreakdown("scale nonfinite") }
            return value
        }
    }
}

public enum NXFGMRES {
    /// Flexible restarted GMRES. The operator is never materialized. Modified Gram-Schmidt uses
    /// one reorthogonalization pass when loss of orthogonality is visible. The preconditioned
    /// basis Z is retained separately from Krylov basis V, so nonlinear/local patch smoothers are legal.
    public static func solve(operator A: any NXLinearOperator,
                             preconditioner M: any NXPreconditioner,
                             rhs b: [Float], initial x0: [Float]? = nil,
                             maximumIterations: Int, restart: Int,
                             relativeTolerance: Float, absoluteTolerance: Float) throws -> NXKrylovResult {
        let n = A.dimension
        guard n > 0, M.dimension == n, b.count == n,
              maximumIterations > 0, restart > 0, restart <= maximumIterations,
              relativeTolerance > 0, absoluteTolerance > 0,
              b.allSatisfy(\.isFinite) else { throw NXError.invalidConfiguration("FGMRES") }
        var x = x0 ?? Array(repeating: 0, count: n)
        guard x.count == n, x.allSatisfy(\.isFinite) else { throw NXError.invalidState("FGMRES initial guess") }
        let bNorm = try NXVectorMath.norm(b)
        let threshold = max(absoluteTolerance, relativeTolerance * max(bNorm, 1))
        var total = 0

        while total < maximumIterations {
            var ax = Array(repeating: Float.zero, count: n)
            try A.apply(x, into: &ax)
            guard ax.count == n, ax.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("operator output") }
            var r = b
            try NXVectorMath.axpy(alpha: -1, x: ax, y: &r)
            let beta = try NXVectorMath.norm(r)
            if beta <= threshold { return NXKrylovResult(solution: x, iterations: total, residualNorm: beta, converged: true) }

            let m = min(restart, maximumIterations - total)
            var v = Array(repeating: Array(repeating: Float.zero, count: n), count: m + 1)
            var z = Array(repeating: Array(repeating: Float.zero, count: n), count: m)
            var h = Array(repeating: Array(repeating: Float.zero, count: m), count: m + 1)
            var cs = Array(repeating: Float.zero, count: m)
            var sn = Array(repeating: Float.zero, count: m)
            var g = Array(repeating: Float.zero, count: m + 1)
            v[0] = try NXVectorMath.scaled(r, by: 1 / beta)
            g[0] = beta
            var used = 0

            for j in 0..<m {
                try M.apply(v[j], iteration: total + j, into: &z[j])
                guard z[j].count == n, z[j].allSatisfy(\.isFinite) else {
                    throw NXError.numericalBreakdown("preconditioner output")
                }
                var w = Array(repeating: Float.zero, count: n)
                try A.apply(z[j], into: &w)
                guard w.count == n, w.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("operator output") }
                let before = try NXVectorMath.norm(w)
                for i in 0...j {
                    h[i][j] = try NXVectorMath.dot(w, v[i])
                    try NXVectorMath.axpy(alpha: -h[i][j], x: v[i], y: &w)
                }
                let after = try NXVectorMath.norm(w)
                if after < 0.5 * before && after > 0 {
                    for i in 0...j {
                        let correction = try NXVectorMath.dot(w, v[i])
                        h[i][j] += correction
                        try NXVectorMath.axpy(alpha: -correction, x: v[i], y: &w)
                    }
                }
                h[j + 1][j] = try NXVectorMath.norm(w)
                if h[j + 1][j] > Float.leastNormalMagnitude {
                    v[j + 1] = try NXVectorMath.scaled(w, by: 1 / h[j + 1][j])
                }

                if j > 0 {
                    for i in 0..<j {
                        let a = h[i][j], b = h[i + 1][j]
                        h[i][j] = cs[i] * a + sn[i] * b
                        h[i + 1][j] = -sn[i] * a + cs[i] * b
                    }
                }
                let a = h[j][j], b2 = h[j + 1][j]
                let denom = hypot(a, b2)
                if denom <= Float.leastNormalMagnitude {
                    cs[j] = 1; sn[j] = 0
                } else {
                    cs[j] = a / denom; sn[j] = b2 / denom
                }
                h[j][j] = cs[j] * a + sn[j] * b2
                h[j + 1][j] = 0
                let gj = g[j]
                g[j] = cs[j] * gj
                g[j + 1] = -sn[j] * gj
                used = j + 1
                total += 1
                if abs(g[j + 1]) <= threshold || total >= maximumIterations { break }
            }

            guard used > 0 else { throw NXError.numericalBreakdown("empty Krylov cycle") }
            var y = Array(repeating: Float.zero, count: used)
            for row in stride(from: used - 1, through: 0, by: -1) {
                var rhs = g[row]
                if row + 1 < used {
                    for col in (row + 1)..<used { rhs -= h[row][col] * y[col] }
                }
                let diagonal = h[row][row]
                guard diagonal.isFinite, abs(diagonal) > Float.leastNormalMagnitude else {
                    throw NXError.numericalBreakdown("singular Krylov Hessenberg")
                }
                y[row] = rhs / diagonal
            }
            for j in 0..<used { try NXVectorMath.axpy(alpha: y[j], x: z[j], y: &x) }
        }

        var ax = Array(repeating: Float.zero, count: n)
        try A.apply(x, into: &ax)
        var residual = b
        try NXVectorMath.axpy(alpha: -1, x: ax, y: &residual)
        let norm = try NXVectorMath.norm(residual)
        return NXKrylovResult(solution: x, iterations: total, residualNorm: norm, converged: norm <= threshold)
    }
}
