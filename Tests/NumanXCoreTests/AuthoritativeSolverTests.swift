import XCTest
@testable import NumanXCore

final class AuthoritativeSolverTests: XCTestCase {
    func testPreparedShadowDoesNotPublishUntilExactTokenCommit() async throws {
        let layout = try NXStateLayout(primalCount: 1)
        let initial = try NXAuthoritativeState(layout: layout, scalars: [2], generation: 9, timeNanoseconds: 100)
        let owner = try NXAuthoritativeSolver(environmentIdentifier: 7, configurationFingerprint: 3,
            topologyFingerprint: 4, initialState: initial)
        let token = try await owner.begin(targetTimeNanoseconds: 200)
        let prepared = try await owner.solve(token: token, profile: .scientific) { state, end in
            QuadraticRootProblem()
        }
        XCTAssertEqual(prepared.state.generation, 10)
        XCTAssertEqual(prepared.state.timeNanoseconds, 200)
        XCTAssertEqual((await owner.committedState()).generation, 9)
        let committed = try await owner.publish(token)
        XCTAssertEqual(committed.generation, 10)
        XCTAssertEqual(committed.timeNanoseconds, 200)
        XCTAssertEqual(committed.scalars[0], sqrt(2), accuracy: 1e-4)
    }

    func testAbortLeavesCommittedBytesUnchanged() async throws {
        let layout = try NXStateLayout(primalCount: 1)
        let initial = try NXAuthoritativeState(layout: layout, scalars: [2], generation: 5, timeNanoseconds: 10)
        let owner = try NXAuthoritativeSolver(environmentIdentifier: 0, configurationFingerprint: 11,
            topologyFingerprint: 12, initialState: initial)
        let token = try await owner.begin(targetTimeNanoseconds: 20)
        _ = try await owner.solve(token: token, profile: .scientific) { _, _ in QuadraticRootProblem() }
        try await owner.abort(token)
        let after = await owner.committedState()
        XCTAssertEqual(after, initial)
    }
}

private struct RootJacobian: NXLinearOperator {
    var dimension: Int { 1 }
    let x: Float
    func apply(_ input: [Float], into output: inout [Float]) throws { output = [2 * x * input[0]] }
}

private struct QuadraticRootProblem: NXNonlinearProblem {
    var dimension: Int { 1 }
    func residual(at state: [Float], into residual: inout [Float]) throws { residual = [state[0] * state[0] - 2] }
    func linearization(at state: [Float]) throws -> any NXLinearOperator { RootJacobian(x: state[0]) }
    func preconditioner(at state: [Float], newtonIteration: Int) throws -> any NXPreconditioner {
        NXIdentityPreconditioner(dimension: 1)
    }
    func safeStep(from state: [Float], along direction: [Float]) throws -> NXGeometricCertificate {
        NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1, minimumVolume: 1,
            minimumRodLength: 1, finite: true, safeStep: 1)
    }
    func diagnostics(at state: [Float], residual: [Float], newtonIterations: Int,
                     krylovIterations: Int, geometric: NXGeometricCertificate) throws -> NXConvergenceCertificate {
        let n = abs(residual[0])
        return NXConvergenceCertificate(residualNorm: n, relativeResidual: n, coneDistance: 0,
            complementarity: 0, pressureResidual: 0, equalityResidual: 0, momentumResidual: n,
            newtonIterations: newtonIterations, krylovIterations: krylovIterations, geometric: geometric)
    }
}
