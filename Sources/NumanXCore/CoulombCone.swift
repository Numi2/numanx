import Foundation

public struct NXCoulombContact: Codable, Equatable, Sendable {
    public var frictionCoefficient: Float
    public var normalCompliance: Float
    public var tangentialCompliance: Float
    public var restitutionVelocity: Float
    public var stabilizationVelocity: Float

    public init(frictionCoefficient: Float,
                normalCompliance: Float = 0,
                tangentialCompliance: Float = 0,
                restitutionVelocity: Float = 0,
                stabilizationVelocity: Float = 0) throws {
        guard frictionCoefficient.isFinite, frictionCoefficient >= 0,
              normalCompliance.isFinite, normalCompliance >= 0,
              tangentialCompliance.isFinite, tangentialCompliance >= 0,
              restitutionVelocity.isFinite, stabilizationVelocity.isFinite else {
            throw NXError.invalidConfiguration("Coulomb contact")
        }
        self.frictionCoefficient = frictionCoefficient
        self.normalCompliance = normalCompliance
        self.tangentialCompliance = tangentialCompliance
        self.restitutionVelocity = restitutionVelocity
        self.stabilizationVelocity = stabilizationVelocity
    }
}

public enum NXLorentzCone {
    /// Euclidean projection onto L3 = {(s,x,y): s >= sqrt(x²+y²)}.
    public static func project(_ q: SIMD3<Float>) throws -> SIMD3<Float> {
        guard q.x.isFinite, q.y.isFinite, q.z.isFinite else {
            throw NXError.invalidState("nonfinite cone vector")
        }
        let t = q.x
        let r = hypot(q.y, q.z)
        if r <= t { return q }
        if r <= -t { return .zero }
        if r == 0 { return SIMD3<Float>(max(t, 0), 0, 0) }
        let scale = 0.5 * (1 + t / r)
        return SIMD3<Float>(0.5 * (r + t), scale * q.y, scale * q.z)
    }

    public static func distance(_ q: SIMD3<Float>) throws -> Float {
        let p = try project(q)
        return simdLength(q - p)
    }

    public static func isInside(_ q: SIMD3<Float>, tolerance: Float = 0) -> Bool {
        q.x.isFinite && q.y.isFinite && q.z.isFinite && tolerance >= 0 &&
        q.x + tolerance >= hypot(q.y, q.z)
    }

    private static func simdLength(_ q: SIMD3<Float>) -> Float {
        sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
    }
}

public enum NXCoulombConeResidual {
    /// Maps physical lambda=(normal,t1,t2) to Lorentz coordinates. For mu=0 the tangential
    /// coordinates are forced to zero and normal contact remains a one-dimensional half-line.
    public static func toCone(_ lambda: SIMD3<Float>, frictionCoefficient mu: Float) throws -> SIMD3<Float> {
        guard lambda.x.isFinite, lambda.y.isFinite, lambda.z.isFinite, mu.isFinite, mu >= 0 else {
            throw NXError.invalidState("contact multiplier")
        }
        if mu == 0 { return SIMD3<Float>(lambda.x, 0, 0) }
        return SIMD3<Float>(lambda.x, lambda.y / mu, lambda.z / mu)
    }

    public static func fromCone(_ q: SIMD3<Float>, frictionCoefficient mu: Float) throws -> SIMD3<Float> {
        guard q.x.isFinite, q.y.isFinite, q.z.isFinite, mu.isFinite, mu >= 0 else {
            throw NXError.invalidState("cone multiplier")
        }
        if mu == 0 { return SIMD3<Float>(q.x, 0, 0) }
        return SIMD3<Float>(q.x, q.y * mu, q.z * mu)
    }

    /// Natural residual R = lambda_hat - Proj_L(lambda_hat - rho*u_hat).
    /// u=(normal relative velocity, tangent1, tangent2) is mapped so positive normal velocity is
    /// separating. Compliance is added to relative velocity before the projection.
    public static func naturalResidual(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                       law: NXCoulombContact, rho: Float) throws -> SIMD3<Float> {
        guard rho.isFinite, rho > 0, u.x.isFinite, u.y.isFinite, u.z.isFinite else {
            throw NXError.invalidConfiguration("contact residual step")
        }
        let corrected = SIMD3<Float>(
            u.x + law.normalCompliance * lambda.x + law.restitutionVelocity + law.stabilizationVelocity,
            u.y + law.tangentialCompliance * lambda.y,
            u.z + law.tangentialCompliance * lambda.z
        )
        let lhat = try toCone(lambda, frictionCoefficient: law.frictionCoefficient)
        let uhat: SIMD3<Float>
        if law.frictionCoefficient == 0 {
            uhat = SIMD3<Float>(corrected.x, 0, 0)
        } else {
            // Dual scaling preserves lambda·u under lambda_t/mu and mu*u_t.
            uhat = SIMD3<Float>(corrected.x,
                                law.frictionCoefficient * corrected.y,
                                law.frictionCoefficient * corrected.z)
        }
        let projected = try NXLorentzCone.project(lhat - rho * uhat)
        let residualHat = lhat - projected
        return try fromCone(residualHat, frictionCoefficient: law.frictionCoefficient)
    }

    public static func diagnostics(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                   law: NXCoulombContact) throws -> (coneDistance: Float, complementarity: Float) {
        let lhat = try toCone(lambda, frictionCoefficient: law.frictionCoefficient)
        let coneDistance = try NXLorentzCone.distance(lhat)
        let normalVelocity = u.x + law.normalCompliance * lambda.x + law.restitutionVelocity + law.stabilizationVelocity
        let tangentPower = lambda.y * (u.y + law.tangentialCompliance * lambda.y)
                         + lambda.z * (u.z + law.tangentialCompliance * lambda.z)
        let complementarity = abs(lambda.x * normalVelocity + tangentPower)
        guard coneDistance.isFinite, complementarity.isFinite else {
            throw NXError.numericalBreakdown("contact diagnostics")
        }
        return (coneDistance, complementarity)
    }
}
