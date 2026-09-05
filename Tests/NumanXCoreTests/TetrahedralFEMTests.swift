import XCTest
@testable import NumanXCore

final class TetrahedralFEMTests: XCTestCase {
    private func element() throws -> NXTetrahedron {
        try NXTetrahedron(
            velocityOffsets: SIMD4<Int32>(0, 3, 6, 9),
            previousPositions: [
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1)
            ],
            inverseRestMatrix: .identity,
            restVolume: 1.0 / 6.0,
            material: NXNeoHookeanMaterial(shearModulus: 2, lameLambda: 3, minimumDeterminant: 0.05)
        )
    }

    func testRestConfigurationHasZeroInternalResidual() throws {
        let fem = try NXTetrahedralFEMContribution(stateDimension: 12, timeStepSeconds: 0.01,
            elements: [element()])
        var residual = Array(repeating: Float.zero, count: 12)
        try fem.addResidual(state: Array(repeating: 0, count: 12), to: &residual)
        XCTAssertLessThan(try NXVectorMath.norm(residual), 1e-6)
    }

    func testAnalyticalTangentMatchesDirectionalFiniteDifference() throws {
        let fem = try NXTetrahedralFEMContribution(stateDimension: 12, timeStepSeconds: 0.02,
            elements: [element()])
        let state: [Float] = [0,0,0, 0.1,0.02,0, 0,0.05,0.01, 0.02,0,0.03]
        let direction: [Float] = [0.03,-0.01,0.02, -0.02,0.04,0, 0.01,0,-0.03, -0.01,0.02,0.01]
        var jv = Array(repeating: Float.zero, count: 12)
        try fem.addJacobianVector(state: state, vector: direction, to: &jv)
        let epsilon: Float = 1e-3
        var plus = state, minus = state
        for i in state.indices { plus[i] += epsilon * direction[i]; minus[i] -= epsilon * direction[i] }
        var rp = Array(repeating: Float.zero, count: 12)
        var rm = Array(repeating: Float.zero, count: 12)
        try fem.addResidual(state: plus, to: &rp)
        try fem.addResidual(state: minus, to: &rm)
        for i in state.indices {
            XCTAssertEqual(jv[i], (rp[i] - rm[i]) / (2 * epsilon), accuracy: 2e-3)
        }
    }

    func testSafeStepLimitsDirectionThatWouldInvertTetrahedron() throws {
        let fem = try NXTetrahedralFEMContribution(stateDimension: 12, timeStepSeconds: 1,
            elements: [element()])
        let state = Array(repeating: Float.zero, count: 12)
        var direction = Array(repeating: Float.zero, count: 12)
        direction[3] = -2 // Move x1 through and past x0 at alpha=1.
        let certificate = try fem.admissibleStep(state: state, direction: direction)
        XCTAssertGreaterThan(certificate.safeStep, 0)
        XCTAssertLessThan(certificate.safeStep, 0.5)
        XCTAssertGreaterThan(certificate.minimumDeterminantF, 0.05)
        XCTAssertGreaterThan(certificate.minimumVolume, 0)
    }
}
