import Foundation

public protocol NXNonlinearProblem: Sendable {
    var dimension: Int { get }
    func residual(at state: [Float], into residual: inout [Float]) throws
    func linearization(at state: [Float]) throws -> any NXLinearOperator
    func preconditioner(at state: [Float], newtonIteration: Int) throws -> any NXPreconditioner
    /// Returns the largest admissible step in (0,1] allowed by CCD, det(F), volume and rod-length checks.
    func safeStep(from state: [Float], along direction: [Float]) throws -> NXGeometricCertificate
    /// Authoritative convergence decomposition. A solver success requires all declared components,
    /// not only a small aggregate residual.
    func diagnostics(at state: [Float], residual: [Float], newtonIterations: Int,
                     krylovIterations: Int, geometric: NXGeometricCertificate) throws -> NXConvergenceCertificate
}

public struct NXNewtonResult: Sendable {
    public var state: [Float]
    public var disposition: NXSolveDisposition
}

public enum NXSemismoothNewton {
    /// Globalized matrix-free semismooth Newton. Candidate state is never published by this routine;
    /// the owner decides whether an accepted shadow generation becomes authoritative.
    public static func solve(problem: any NXNonlinearProblem, initialState: [Float],
                             profile: NXExecutionProfile) throws -> NXNewtonResult {
        let n = problem.dimension
        guard n > 0, initialState.count == n, initialState.allSatisfy(\.isFinite) else {
            return NXNewtonResult(state: initialState, disposition: .invalidInput("initial nonlinear state"))
        }
        var x = initialState
        var residual = Array(repeating: Float.zero, count: n)
        try problem.residual(at: x, into: &residual)
        try validate(residual, count: n, label: "initial residual")
        let initialNorm = max(try NXVectorMath.norm(residual), profile.absoluteResidualTolerance)
        var totalKrylov = 0
        var lastCertificate: NXConvergenceCertificate?

        for newton in 0..<profile.maximumNewtonIterations {
            let residualNorm = try NXVectorMath.norm(residual)
            let relative = residualNorm / initialNorm
            let geometric = try problem.safeStep(from: x, along: Array(repeating: 0, count: n))
            let certificate = try problem.diagnostics(at: x, residual: residual,
                newtonIterations: newton, krylovIterations: totalKrylov, geometric: geometric)
            lastCertificate = certificate
            if convergenceSatisfied(certificate: certificate, residualNorm: residualNorm,
                                    relative: relative, profile: profile) {
                return NXNewtonResult(state: x, disposition: .committed(certificate))
            }

            let jacobian = try problem.linearization(at: x)
            let preconditioner = try problem.preconditioner(at: x, newtonIteration: newton)
            guard jacobian.dimension == n, preconditioner.dimension == n else {
                throw NXError.invalidConfiguration("linearized operator dimension")
            }
            let rhs = try residual.map { value -> Float in
                let negated = -value
                guard negated.isFinite else { throw NXError.numericalBreakdown("Newton RHS") }
                return negated
            }
            let krylov = try NXFGMRES.solve(operator: jacobian, preconditioner: preconditioner,
                rhs: rhs, maximumIterations: profile.maximumKrylovIterations,
                restart: profile.krylovRestart,
                relativeTolerance: min(0.5, max(profile.relativeResidualTolerance, 0.05 * relative)),
                absoluteTolerance: profile.absoluteResidualTolerance)
            totalKrylov += krylov.iterations
            guard krylov.solution.count == n, krylov.solution.allSatisfy(\.isFinite) else {
                throw NXError.numericalBreakdown("Newton direction")
            }

            let safety = try problem.safeStep(from: x, along: krylov.solution)
                .validated(minimumSafeStep: profile.minimumSafeStep)
            var alpha = min(1, safety.safeStep)
            let merit0 = 0.5 * residualNorm * residualNorm
            var accepted = false
            var acceptedState = x
            var acceptedResidual = residual

            for _ in 0..<profile.maximumLineSearchTrials {
                if alpha < profile.minimumSafeStep { break }
                var candidate = x
                try NXVectorMath.axpy(alpha: alpha, x: krylov.solution, y: &candidate)
                guard candidate.allSatisfy(\.isFinite) else {
                    alpha *= 0.5
                    continue
                }
                var candidateResidual = Array(repeating: Float.zero, count: n)
                do {
                    try problem.residual(at: candidate, into: &candidateResidual)
                    try validate(candidateResidual, count: n, label: "line-search residual")
                } catch {
                    alpha *= 0.5
                    continue
                }
                let candidateNorm = try NXVectorMath.norm(candidateResidual)
                let merit = 0.5 * candidateNorm * candidateNorm
                let required = merit0 * max(0, 1 - profile.armijo * alpha)
                if merit <= required || candidateNorm <= profile.absoluteResidualTolerance {
                    accepted = true
                    acceptedState = candidate
                    acceptedResidual = candidateResidual
                    break
                }
                alpha *= 0.5
            }

            guard accepted else {
                return NXNewtonResult(state: x,
                    disposition: profile.mode == .boundedDeterministic
                        ? .substepRequired(lastCertificate) : .rejected(lastCertificate))
            }
            x = acceptedState
            residual = acceptedResidual
        }

        let residualNorm = try NXVectorMath.norm(residual)
        let relative = residualNorm / initialNorm
        let geometric = try problem.safeStep(from: x, along: Array(repeating: 0, count: n))
        let certificate = try problem.diagnostics(at: x, residual: residual,
            newtonIterations: profile.maximumNewtonIterations, krylovIterations: totalKrylov, geometric: geometric)
        if convergenceSatisfied(certificate: certificate, residualNorm: residualNorm,
                                relative: relative, profile: profile) {
            return NXNewtonResult(state: x, disposition: .committed(certificate))
        }
        return NXNewtonResult(state: x,
            disposition: profile.mode == .boundedDeterministic
                ? .substepRequired(certificate) : .rejected(certificate))
    }

    private static func convergenceSatisfied(certificate: NXConvergenceCertificate,
                                             residualNorm: Float, relative: Float,
                                             profile: NXExecutionProfile) -> Bool {
        let residualPass = residualNorm <= profile.absoluteResidualTolerance
            || relative <= profile.relativeResidualTolerance
        return residualPass
            && certificate.residualNorm.isFinite
            && certificate.relativeResidual.isFinite
            && certificate.coneDistance.isFinite
            && certificate.complementarity.isFinite
            && certificate.pressureResidual.isFinite
            && certificate.equalityResidual.isFinite
            && certificate.momentumResidual.isFinite
            && certificate.coneDistance <= profile.coneTolerance
            && certificate.complementarity <= profile.complementarityTolerance
            && certificate.geometric.finite
            && certificate.geometric.safeStep >= 0
            && certificate.geometric.safeStep <= 1
    }

    private static func validate(_ vector: [Float], count: Int, label: String) throws {
        guard vector.count == count, vector.allSatisfy(\.isFinite) else {
            throw NXError.numericalBreakdown(label)
        }
    }
}
