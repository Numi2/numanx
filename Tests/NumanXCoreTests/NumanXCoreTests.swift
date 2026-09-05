import XCTest
@testable import NumanXCore

final class NumanXCoreTests: XCTestCase {
    func testLorentzProjectionCases() throws {
        XCTAssertEqual(try NXLorentzCone.project(SIMD3<Float>(2, 1, 0)), SIMD3<Float>(2, 1, 0))
        XCTAssertEqual(try NXLorentzCone.project(SIMD3<Float>(-2, 1, 0)), .zero)
        let p = try NXLorentzCone.project(SIMD3<Float>(0, 2, 0))
        XCTAssertEqual(p.x, 1, accuracy: 1e-6)
        XCTAssertEqual(p.y, 1, accuracy: 1e-6)
        XCTAssertTrue(NXLorentzCone.isInside(p, tolerance: 1e-6))
    }

    func testCoulombResidualVanishesForSeparatingZeroMultiplier() throws {
        let law = try NXCoulombContact(frictionCoefficient: 0.7)
        let r = try NXCoulombConeResidual.naturalResidual(lambda: .zero,
            relativeVelocity: SIMD3<Float>(1, 0.2, -0.1), law: law, rho: 1)
        XCTAssertEqual(r.x, 0, accuracy: 1e-6)
        XCTAssertEqual(r.y, 0, accuracy: 1e-6)
        XCTAssertEqual(r.z, 0, accuracy: 1e-6)
    }

    func testFGMRESSolvesSmallNonsymmetricSystem() throws {
        let a = DenseOperator(matrix: [4, 1, 0, 1, 3, 1, 0, -1, 2], dimension: 3)
        let b: [Float] = [1, 2, 3]
        let result = try NXFGMRES.solve(operator: a, preconditioner: NXIdentityPreconditioner(dimension: 3),
            rhs: b, maximumIterations: 12, restart: 3, relativeTolerance: 1e-6, absoluteTolerance: 1e-7)
        XCTAssertTrue(result.converged)
        var ax = Array(repeating: Float.zero, count: 3)
        try a.apply(result.solution, into: &ax)
        for i in 0..<3 { XCTAssertEqual(ax[i], b[i], accuracy: 1e-4) }
    }

    func testTensorSchurAveragesOverlappingCorrections() throws {
        let p0 = try NXPatchDescriptor(identifier: 0, patchClass: .micro, physicsBlock: .rigid,
            unknownIndices: [0, 1])
        let p1 = try NXPatchDescriptor(identifier: 1, patchClass: .micro, physicsBlock: .contact,
            unknownIndices: [1, 2])
        let local = NXDensePatchSolver(matrices: [0: [1, 0, 0, 1], 1: [1, 0, 0, 1]])
        let pre = try NXTensorSchurPreconditioner(dimension: 3, patches: [p0, p1],
            policy: NXMixedPrecisionPolicy(), localSolver: local)
        var correction: [Float] = []
        try pre.apply([1, 2, 3], iteration: 0, into: &correction)
        XCTAssertEqual(correction, [1, 2, 3])
    }

    func testSemismoothNewtonConvergesScalarQuadratic() throws {
        let problem = QuadraticProblem()
        let result = try NXSemismoothNewton.solve(problem: problem, initialState: [2],
            profile: try NXExecutionProfile(mode: .adaptiveAccuracy, maximumNewtonIterations: 10,
                maximumKrylovIterations: 4, krylovRestart: 2, maximumLineSearchTrials: 5,
                relativeResidualTolerance: 1e-6, absoluteResidualTolerance: 1e-6,
                coneTolerance: 1e-6, complementarityTolerance: 1e-6))
        switch result.disposition {
        case .committed:
            XCTAssertEqual(result.state[0], sqrt(2), accuracy: 1e-4)
        default: XCTFail("expected committed Newton solution")
        }
    }

    func testStateLayoutContactTriplesAreExact() throws {
        let layout = try NXStateLayout(primalCount: 6, pressureCount: 2, equalityMultiplierCount: 1, contactCount: 4)
        XCTAssertEqual(layout.scalarCount, 21)
        XCTAssertEqual(layout.contactCount, 4)
        XCTAssertEqual(layout.contactMultiplierRange.count, 12)
    }
}

private struct DenseOperator: NXLinearOperator {
    let matrix: [Float]
    let dimension: Int
    func apply(_ x: [Float], into y: inout [Float]) throws {
        guard matrix.count == dimension * dimension, x.count == dimension else { throw NXError.invalidState("dense operator") }
        y = Array(repeating: 0, count: dimension)
        for row in 0..<dimension {
            for column in 0..<dimension { y[row] += matrix[row * dimension + column] * x[column] }
        }
    }
}

private struct QuadraticJacobian: NXLinearOperator {
    let x: Float
    var dimension: Int { 1 }
    func apply(_ input: [Float], into output: inout [Float]) throws { output = [2 * x * input[0]] }
}

private struct QuadraticProblem: NXNonlinearProblem {
    var dimension: Int { 1 }
    func residual(at state: [Float], into residual: inout [Float]) throws { residual = [state[0] * state[0] - 2] }
    func linearization(at state: [Float]) throws -> any NXLinearOperator { QuadraticJacobian(x: state[0]) }
    func preconditioner(at state: [Float], newtonIteration: Int) throws -> any NXPreconditioner {
        NXIdentityPreconditioner(dimension: 1)
    }
    func safeStep(from state: [Float], along direction: [Float]) throws -> NXGeometricCertificate {
        NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1, minimumVolume: 1,
            minimumRodLength: 1, finite: true, safeStep: 1)
    }
    func diagnostics(at state: [Float], residual: [Float], newtonIterations: Int,
                     krylovIterations: Int, geometric: NXGeometricCertificate) throws -> NXConvergenceCertificate {
        let norm = abs(residual[0])
        return NXConvergenceCertificate(residualNorm: norm, relativeResidual: norm,
            coneDistance: 0, complementarity: 0, pressureResidual: 0, equalityResidual: 0,
            momentumResidual: norm, newtonIterations: newtonIterations,
            krylovIterations: krylovIterations, geometric: geometric)
    }
}
