import Foundation

public struct NXRodSegment: Codable, Equatable, Sendable {
    public var firstVelocityOffset: Int
    public var secondVelocityOffset: Int
    public var firstPreviousPosition: SIMD3<Float>
    public var secondPreviousPosition: SIMD3<Float>
    public var restLength: Float
    public var axialStiffness: Float
    public var minimumLength: Float

    public init(firstVelocityOffset: Int, secondVelocityOffset: Int,
                firstPreviousPosition: SIMD3<Float>, secondPreviousPosition: SIMD3<Float>,
                restLength: Float, axialStiffness: Float, minimumLength: Float = 1e-5) throws {
        guard firstVelocityOffset >= 0, secondVelocityOffset >= 0,
              firstVelocityOffset != secondVelocityOffset,
              firstPreviousPosition.x.isFinite, firstPreviousPosition.y.isFinite, firstPreviousPosition.z.isFinite,
              secondPreviousPosition.x.isFinite, secondPreviousPosition.y.isFinite, secondPreviousPosition.z.isFinite,
              restLength.isFinite, restLength > 0, axialStiffness.isFinite, axialStiffness > 0,
              minimumLength.isFinite, minimumLength > 0, minimumLength < restLength else {
            throw NXError.invalidConfiguration("rod segment")
        }
        self.firstVelocityOffset = firstVelocityOffset; self.secondVelocityOffset = secondVelocityOffset
        self.firstPreviousPosition = firstPreviousPosition; self.secondPreviousPosition = secondPreviousPosition
        self.restLength = restLength; self.axialStiffness = axialStiffness; self.minimumLength = minimumLength
    }
}

/// Implicit axial discrete-rod energy 0.5*k*(|x1-x0|-L)^2. Bending/twist are separate blocks so
/// sutures/cables can choose Kirchhoff/DER material frames without changing the axial authority.
public struct NXDiscreteRodStretchContribution: NXPhysicsContribution {
    public let name = "discrete-rod-stretch"
    public let stateDimension: Int
    public let timeStepSeconds: Float
    public let segments: [NXRodSegment]

    public init(stateDimension: Int, timeStepSeconds: Float, segments: [NXRodSegment]) throws {
        guard stateDimension > 0, timeStepSeconds.isFinite, timeStepSeconds > 0, !segments.isEmpty else {
            throw NXError.invalidConfiguration("discrete rods")
        }
        for s in segments {
            guard s.firstVelocityOffset <= stateDimension - 3,
                  s.secondVelocityOffset <= stateDimension - 3 else {
                throw NXError.invalidConfiguration("rod velocity range")
            }
        }
        self.stateDimension = stateDimension; self.timeStepSeconds = timeStepSeconds; self.segments = segments
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("rod residual")
        }
        for s in segments {
            let (edge, length) = try geometry(s, state: state)
            let n = edge / length
            let force = timeStepSeconds * s.axialStiffness * (length - s.restLength) * n
            add(-force, offset: s.firstVelocityOffset, to: &residual)
            add(force, offset: s.secondVelocityOffset, to: &residual)
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("rod residual nonfinite") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension,
              state.allSatisfy(\.isFinite), vector.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("rod Jv")
        }
        for s in segments {
            let (edge, length) = try geometry(s, state: state)
            let n = edge / length
            let dv = value(vector, offset: s.secondVelocityOffset) - value(vector, offset: s.firstVelocityOffset)
            let ratio = s.restLength / length
            let projection = n * dot(n, dv)
            let hDv = s.axialStiffness * ((1 - ratio) * dv + ratio * projection)
            let value = timeStepSeconds * timeStepSeconds * hDv
            add(-value, offset: s.firstVelocityOffset, to: &product)
            add(value, offset: s.secondVelocityOffset, to: &product)
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("rod Jv nonfinite") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("rod safe step")
        }
        var safe: Float = 1
        var minimum = Float.greatestFiniteMagnitude
        for s in segments {
            let (_, currentLength) = try geometry(s, state: state)
            guard currentLength > s.minimumLength else {
                return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
                    minimumVolume: 1, minimumRodLength: currentLength, finite: currentLength.isFinite, safeStep: 0)
            }
            let full = length(s, state: state, direction: direction, alpha: 1)
            if !full.isFinite || full <= s.minimumLength {
                var low: Float = 0, high: Float = safe
                for _ in 0..<28 {
                    let mid = 0.5 * (low + high)
                    let l = length(s, state: state, direction: direction, alpha: mid)
                    if l.isFinite && l > s.minimumLength { low = mid } else { high = mid }
                }
                safe = min(safe, low * 0.95)
            }
        }
        for s in segments { minimum = min(minimum, length(s, state: state, direction: direction, alpha: safe)) }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: minimum, finite: minimum.isFinite, safeStep: safe)
    }

    private func geometry(_ s: NXRodSegment, state: [Float]) throws -> (SIMD3<Float>, Float) {
        let e = position(s, second: true, state: state) - position(s, second: false, state: state)
        let l = sqrt(dot(e, e))
        guard l.isFinite, l > s.minimumLength else { throw NXError.geometryRejected }
        return (e, l)
    }

    private func length(_ s: NXRodSegment, state: [Float], direction: [Float], alpha: Float) -> Float {
        let x0 = position(s, second: false, state: state)
            + timeStepSeconds * alpha * value(direction, offset: s.firstVelocityOffset)
        let x1 = position(s, second: true, state: state)
            + timeStepSeconds * alpha * value(direction, offset: s.secondVelocityOffset)
        let e = x1 - x0
        return sqrt(dot(e, e))
    }

    private func position(_ s: NXRodSegment, second: Bool, state: [Float]) -> SIMD3<Float> {
        let o = second ? s.secondVelocityOffset : s.firstVelocityOffset
        let previous = second ? s.secondPreviousPosition : s.firstPreviousPosition
        return previous + timeStepSeconds * value(state, offset: o)
    }

    private func value(_ vector: [Float], offset: Int) -> SIMD3<Float> {
        SIMD3<Float>(vector[offset], vector[offset + 1], vector[offset + 2])
    }
    private func add(_ v: SIMD3<Float>, offset: Int, to vector: inout [Float]) {
        vector[offset] += v.x; vector[offset + 1] += v.y; vector[offset + 2] += v.z
    }
    private func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float { a.x*b.x + a.y*b.y + a.z*b.z }
}
