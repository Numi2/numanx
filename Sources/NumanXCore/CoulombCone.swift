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

    /// One element of the Clarke generalized derivative away from the cone apex. At the two region
    /// boundaries the selected branch is deterministic, which is sufficient for semismooth Newton.
    public static func derivative(at q: SIMD3<Float>, appliedTo dq: SIMD3<Float>) throws -> SIMD3<Float> {
        guard q.x.isFinite, q.y.isFinite, q.z.isFinite,
              dq.x.isFinite, dq.y.isFinite, dq.z.isFinite else {
            throw NXError.invalidState("cone derivative")
        }
        let t = q.x
        let v = SIMD2<Float>(q.y, q.z)
        let dv = SIMD2<Float>(dq.y, dq.z)
        let r = hypot(v.x, v.y)
        if r <= t { return dq }
        if r <= -t { return .zero }
        guard r > Float.leastNormalMagnitude else {
            return SIMD3<Float>(max(dq.x, 0), 0, 0)
        }
        let dr = (v.x * dv.x + v.y * dv.y) / r
        let dp0 = 0.5 * (dq.x + dr)
        let scale = 0.5 * (1 + t / r)
        let dscale = 0.5 * (dq.x / r - t * dr / (r * r))
        let dpv = scale * dv + dscale * v
        let result = SIMD3<Float>(dp0, dpv.x, dpv.y)
        guard result.x.isFinite, result.y.isFinite, result.z.isFinite else {
            throw NXError.numericalBreakdown("cone derivative nonfinite")
        }
        return result
    }

    public static func distance(_ q: SIMD3<Float>) throws -> Float {
        let p = try project(q)
        return sqrt((q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y) + (q.z - p.z) * (q.z - p.z))
    }

    public static func isInside(_ q: SIMD3<Float>, tolerance: Float = 0) -> Bool {
        q.x.isFinite && q.y.isFinite && q.z.isFinite && tolerance >= 0 &&
        q.x + tolerance >= hypot(q.y, q.z)
    }
}

public enum NXCoulombConeResidual {
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

    private static func correctedVelocity(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                          law: NXCoulombContact) -> SIMD3<Float> {
        SIMD3<Float>(u.x + law.normalCompliance * lambda.x + law.restitutionVelocity + law.stabilizationVelocity,
                     u.y + law.tangentialCompliance * lambda.y,
                     u.z + law.tangentialCompliance * lambda.z)
    }

    private static func dualVelocity(_ u: SIMD3<Float>, mu: Float) -> SIMD3<Float> {
        mu == 0 ? SIMD3<Float>(u.x, 0, 0) : SIMD3<Float>(u.x, mu * u.y, mu * u.z)
    }

    public static func naturalResidual(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                       law: NXCoulombContact, rho: Float) throws -> SIMD3<Float> {
        guard rho.isFinite, rho > 0, u.x.isFinite, u.y.isFinite, u.z.isFinite else {
            throw NXError.invalidConfiguration("contact residual step")
        }
        let lhat = try toCone(lambda, frictionCoefficient: law.frictionCoefficient)
        let uhat = dualVelocity(correctedVelocity(lambda: lambda, relativeVelocity: u, law: law),
                                mu: law.frictionCoefficient)
        let projected = try NXLorentzCone.project(lhat - rho * uhat)
        return try fromCone(lhat - projected, frictionCoefficient: law.frictionCoefficient)
    }

    /// Analytical directional derivative of the selected semismooth residual branch. `dVelocity`
    /// is J*dv before compliance; `dLambda` contributes both directly and through compliance.
    public static func directionalDerivative(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                             dLambda: SIMD3<Float>, dVelocity: SIMD3<Float>,
                                             law: NXCoulombContact, rho: Float) throws -> SIMD3<Float> {
        guard rho.isFinite, rho > 0,
              dLambda.x.isFinite, dLambda.y.isFinite, dLambda.z.isFinite,
              dVelocity.x.isFinite, dVelocity.y.isFinite, dVelocity.z.isFinite else {
            throw NXError.invalidConfiguration("contact directional derivative")
        }
        let mu = law.frictionCoefficient
        let lhat = try toCone(lambda, frictionCoefficient: mu)
        let dlhat = try toCone(dLambda, frictionCoefficient: mu)
        let corrected = correctedVelocity(lambda: lambda, relativeVelocity: u, law: law)
        let dcorrected = SIMD3<Float>(dVelocity.x + law.normalCompliance * dLambda.x,
                                      dVelocity.y + law.tangentialCompliance * dLambda.y,
                                      dVelocity.z + law.tangentialCompliance * dLambda.z)
        let uhat = dualVelocity(corrected, mu: mu)
        let duhat = dualVelocity(dcorrected, mu: mu)
        let q = lhat - rho * uhat
        let dq = dlhat - rho * duhat
        let projectedDerivative = try NXLorentzCone.derivative(at: q, appliedTo: dq)
        return try fromCone(dlhat - projectedDerivative, frictionCoefficient: mu)
    }

    public static func diagnostics(lambda: SIMD3<Float>, relativeVelocity u: SIMD3<Float>,
                                   law: NXCoulombContact) throws -> (coneDistance: Float, complementarity: Float) {
        let lhat = try toCone(lambda, frictionCoefficient: law.frictionCoefficient)
        let coneDistance = try NXLorentzCone.distance(lhat)
        let corrected = correctedVelocity(lambda: lambda, relativeVelocity: u, law: law)
        let complementarity = abs(lambda.x * corrected.x + lambda.y * corrected.y + lambda.z * corrected.z)
        guard coneDistance.isFinite, complementarity.isFinite else {
            throw NXError.numericalBreakdown("contact diagnostics")
        }
        return (coneDistance, complementarity)
    }
}
