import Foundation

public struct NXMatrix6: Codable, Equatable, Sendable {
    /// Row-major 36 entries.
    public var values: [Float]

    public init(_ values: [Float]) throws {
        guard values.count == 36, values.allSatisfy(\.isFinite) else {
            throw NXError.invalidConfiguration("6x6 matrix")
        }
        self.values = values
    }

    public static var zero: NXMatrix6 { try! NXMatrix6(Array(repeating: 0, count: 36)) }
    public static var identity: NXMatrix6 {
        var a = Array(repeating: Float.zero, count: 36)
        for i in 0..<6 { a[i * 6 + i] = 1 }
        return try! NXMatrix6(a)
    }

    public subscript(row: Int, column: Int) -> Float {
        get { values[row * 6 + column] }
        set { values[row * 6 + column] = newValue }
    }

    public var transposed: NXMatrix6 {
        var result = NXMatrix6.zero
        for r in 0..<6 { for c in 0..<6 { result[r, c] = self[c, r] } }
        return result
    }

    public func apply(_ x: NXVector6) -> NXVector6 {
        var y = NXVector6.zero
        for r in 0..<6 {
            var value: Float = 0
            for c in 0..<6 { value += self[r, c] * x[c] }
            y[r] = value
        }
        return y
    }

    public func multiplied(by rhs: NXMatrix6) -> NXMatrix6 {
        var result = NXMatrix6.zero
        for r in 0..<6 {
            for c in 0..<6 {
                var value: Float = 0
                for k in 0..<6 { value += self[r, k] * rhs[k, c] }
                result[r, c] = value
            }
        }
        return result
    }

    public static func + (lhs: NXMatrix6, rhs: NXMatrix6) -> NXMatrix6 {
        try! NXMatrix6(zip(lhs.values, rhs.values).map(+))
    }
    public static func - (lhs: NXMatrix6, rhs: NXMatrix6) -> NXMatrix6 {
        try! NXMatrix6(zip(lhs.values, rhs.values).map(-))
    }
    public static func * (lhs: Float, rhs: NXMatrix6) -> NXMatrix6 {
        try! NXMatrix6(rhs.values.map { lhs * $0 })
    }
    public static func outer(_ a: NXVector6, _ b: NXVector6) -> NXMatrix6 {
        var result = NXMatrix6.zero
        for r in 0..<6 { for c in 0..<6 { result[r, c] = a[r] * b[c] } }
        return result
    }
}

public extension NXVector6 {
    static func + (lhs: NXVector6, rhs: NXVector6) -> NXVector6 {
        var result = NXVector6.zero
        for i in 0..<6 { result[i] = lhs[i] + rhs[i] }
        return result
    }
    static func - (lhs: NXVector6, rhs: NXVector6) -> NXVector6 {
        var result = NXVector6.zero
        for i in 0..<6 { result[i] = lhs[i] - rhs[i] }
        return result
    }
    static prefix func - (x: NXVector6) -> NXVector6 {
        var result = NXVector6.zero
        for i in 0..<6 { result[i] = -x[i] }
        return result
    }
    static func * (lhs: Float, rhs: NXVector6) -> NXVector6 {
        var result = NXVector6.zero
        for i in 0..<6 { result[i] = lhs * rhs[i] }
        return result
    }
    func dot(_ rhs: NXVector6) -> Float {
        var value: Float = 0
        for i in 0..<6 { value += self[i] * rhs[i] }
        return value
    }
}
