import Foundation

public struct NXMatrix3: Codable, Equatable, Sendable {
    public var c0: SIMD3<Float>
    public var c1: SIMD3<Float>
    public var c2: SIMD3<Float>

    public init(_ c0: SIMD3<Float>, _ c1: SIMD3<Float>, _ c2: SIMD3<Float>) {
        self.c0 = c0; self.c1 = c1; self.c2 = c2
    }
    public static let identity = NXMatrix3(SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1))
    public static let zero = NXMatrix3(.zero, .zero, .zero)

    public subscript(row: Int, column: Int) -> Float {
        get {
            switch column { case 0: c0[row]; case 1: c1[row]; case 2: c2[row]; default: preconditionFailure("NXMatrix3 column") }
        }
        set {
            switch column { case 0: c0[row] = newValue; case 1: c1[row] = newValue; case 2: c2[row] = newValue; default: preconditionFailure("NXMatrix3 column") }
        }
    }

    public var allFinite: Bool {
        (0..<3).allSatisfy { r in (0..<3).allSatisfy { self[r, $0].isFinite } }
    }

    public var transposed: NXMatrix3 {
        NXMatrix3(SIMD3<Float>(c0.x, c1.x, c2.x),
                  SIMD3<Float>(c0.y, c1.y, c2.y),
                  SIMD3<Float>(c0.z, c1.z, c2.z))
    }

    public var determinant: Float {
        dot(c0, cross(c1, c2))
    }

    public func inverse() throws -> NXMatrix3 {
        let det = determinant
        guard det.isFinite, abs(det) > 1e-12 else { throw NXError.numericalBreakdown("singular 3x3 matrix") }
        // For a column-major matrix, inverse rows are cross(c1,c2), cross(c2,c0), cross(c0,c1) / det.
        let r0 = cross(c1, c2) / det
        let r1 = cross(c2, c0) / det
        let r2 = cross(c0, c1) / det
        return NXMatrix3(SIMD3<Float>(r0.x, r1.x, r2.x),
                         SIMD3<Float>(r0.y, r1.y, r2.y),
                         SIMD3<Float>(r0.z, r1.z, r2.z))
    }

    public func multiplied(by rhs: NXMatrix3) -> NXMatrix3 {
        NXMatrix3(apply(rhs.c0), apply(rhs.c1), apply(rhs.c2))
    }

    public func apply(_ x: SIMD3<Float>) -> SIMD3<Float> {
        c0 * x.x + c1 * x.y + c2 * x.z
    }

    public static func + (lhs: NXMatrix3, rhs: NXMatrix3) -> NXMatrix3 {
        NXMatrix3(lhs.c0 + rhs.c0, lhs.c1 + rhs.c1, lhs.c2 + rhs.c2)
    }
    public static func - (lhs: NXMatrix3, rhs: NXMatrix3) -> NXMatrix3 {
        NXMatrix3(lhs.c0 - rhs.c0, lhs.c1 - rhs.c1, lhs.c2 - rhs.c2)
    }
    public static func * (lhs: Float, rhs: NXMatrix3) -> NXMatrix3 {
        NXMatrix3(lhs * rhs.c0, lhs * rhs.c1, lhs * rhs.c2)
    }

    public static func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> NXMatrix3 {
        NXMatrix3(a * b.x, a * b.y, a * b.z)
    }

    public static func dot(_ a: NXMatrix3, _ b: NXMatrix3) -> Float {
        dot(a.c0, b.c0) + dot(a.c1, b.c1) + dot(a.c2, b.c2)
    }

    @inline(__always) private static func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
    @inline(__always) private static func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x)
    }
    @inline(__always) private func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float { Self.dot(a, b) }
    @inline(__always) private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> { Self.cross(a, b) }
}
