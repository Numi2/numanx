import Foundation

public struct NXIncompressibleTetrahedron: Codable, Equatable, Sendable {
    public var velocityOffsets: SIMD4<Int32>
    public var pressureIndex: Int
    public var previousPositions: [SIMD3<Float>]
    public var inverseRestMatrix: NXMatrix3
    public var restVolume: Float
    public var pressureCompliance: Float
    public var minimumDeterminant: Float

    public init(velocityOffsets: SIMD4<Int32>, pressureIndex: Int,
                previousPositions: [SIMD3<Float>], inverseRestMatrix: NXMatrix3,
                restVolume: Float, pressureCompliance: Float = 0,
                minimumDeterminant: Float = 0.05) throws {
        guard pressureIndex >= 0, previousPositions.count == 4,
              (0..<4).allSatisfy({ velocityOffsets[$0] >= 0 }),
              inverseRestMatrix.allFinite, abs(inverseRestMatrix.determinant) > 1e-12,
              restVolume.isFinite, restVolume > 0,
              pressureCompliance.isFinite, pressureCompliance >= 0,
              minimumDeterminant.isFinite, minimumDeterminant > 0,
              previousPositions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw NXError.invalidConfiguration("incompressible tetrahedron")
        }
        self.velocityOffsets = velocityOffsets; self.pressureIndex = pressureIndex
        self.previousPositions = previousPositions; self.inverseRestMatrix = inverseRestMatrix
        self.restVolume = restVolume; self.pressureCompliance = pressureCompliance
        self.minimumDeterminant = minimumDeterminant
    }
}

/// Mixed displacement/pressure block based on L = -p(J-1)+0.5*C*p².
/// Momentum receives -p*cof(F); pressure residual is -(J-1)+C*p. The directional derivative is
/// analytical, preserving the coupled KKT structure instead of penalty-only incompressibility.
public struct NXIncompressibilityContribution: NXPhysicsContribution {
    public let name = "mixed-tetrahedral-incompressibility"
    public let stateDimension: Int
    public let timeStepSeconds: Float
    public let elements: [NXIncompressibleTetrahedron]

    public init(stateDimension: Int, timeStepSeconds: Float,
                elements: [NXIncompressibleTetrahedron]) throws {
        guard stateDimension > 0, timeStepSeconds.isFinite, timeStepSeconds > 0, !elements.isEmpty else {
            throw NXError.invalidConfiguration("incompressibility contribution")
        }
        var pressure = Set<Int>()
        for element in elements {
            guard element.pressureIndex < stateDimension,
                  pressure.insert(element.pressureIndex).inserted else {
                throw NXError.invalidConfiguration("pressure index")
            }
        }
        for element in elements {
            for lane in 0..<4 {
                let offset = Int(element.velocityOffsets[lane])
                guard offset <= stateDimension - 3,
                      !pressure.contains(offset), !pressure.contains(offset + 1), !pressure.contains(offset + 2) else {
                    throw NXError.invalidConfiguration("pressure overlaps velocity state")
                }
            }
        }
        self.stateDimension = stateDimension; self.timeStepSeconds = timeStepSeconds; self.elements = elements
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("incompressibility residual")
        }
        for element in elements {
            let F = deformationGradient(element, state: state)
            let J = F.determinant
            guard J.isFinite, J > element.minimumDeterminant else { throw NXError.geometryRejected }
            let invT = try F.inverse().transposed
            let cofactor = J * invT
            let p = state[element.pressureIndex]
            let P = -p * cofactor
            let gradients = shapeGradients(element)
            let mechanicalScale = timeStepSeconds * element.restVolume
            for node in 0..<4 {
                let value = mechanicalScale * P.apply(gradients[node])
                let offset = Int(element.velocityOffsets[node])
                residual[offset] += value.x; residual[offset + 1] += value.y; residual[offset + 2] += value.z
            }
            residual[element.pressureIndex] += element.restVolume * (-(J - 1) + element.pressureCompliance * p)
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("incompressibility residual") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension,
              state.allSatisfy(\.isFinite), vector.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("incompressibility Jv")
        }
        for element in elements {
            let F = deformationGradient(element, state: state)
            let J = F.determinant
            guard J.isFinite, J > element.minimumDeterminant else { throw NXError.geometryRejected }
            let inv = try F.inverse(), invT = inv.transposed
            let dFPerVelocity = directionalGradient(element, vector: vector)
            let tracePerVelocity = NXMatrix3.dot(inv.transposed, dFPerVelocity)
            let dJPerVelocity = J * tracePerVelocity
            let cofactor = J * invT
            let dCofactorPerVelocity = dJPerVelocity * invT
                - J * invT.multiplied(by: dFPerVelocity.transposed).multiplied(by: invT)
            let p = state[element.pressureIndex]
            let dp = vector[element.pressureIndex]
            let dPFromPressure = -dp * cofactor
            let dPFromVelocity = -p * dCofactorPerVelocity
            let gradients = shapeGradients(element)
            let pressureScale = timeStepSeconds * element.restVolume
            let velocityScale = timeStepSeconds * timeStepSeconds * element.restVolume
            for node in 0..<4 {
                let value = pressureScale * dPFromPressure.apply(gradients[node])
                          + velocityScale * dPFromVelocity.apply(gradients[node])
                let offset = Int(element.velocityOffsets[node])
                product[offset] += value.x; product[offset + 1] += value.y; product[offset + 2] += value.z
            }
            product[element.pressureIndex] += element.restVolume
                * (-timeStepSeconds * dJPerVelocity + element.pressureCompliance * dp)
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("incompressibility Jv") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("incompressibility safe step")
        }
        var safe: Float = 1
        var minJ = Float.greatestFiniteMagnitude
        var minVolume = Float.greatestFiniteMagnitude
        for element in elements {
            let currentJ = deformationGradient(element, state: state).determinant
            guard currentJ.isFinite, currentJ > element.minimumDeterminant else {
                return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: currentJ,
                    minimumVolume: element.restVolume * currentJ, minimumRodLength: 1,
                    finite: currentJ.isFinite, safeStep: 0)
            }
            let fullJ = deformationGradient(element, state: state, direction: direction, alpha: 1).determinant
            if !fullJ.isFinite || fullJ <= element.minimumDeterminant {
                var low: Float = 0, high: Float = safe
                for _ in 0..<28 {
                    let mid = 0.5 * (low + high)
                    let j = deformationGradient(element, state: state, direction: direction, alpha: mid).determinant
                    if j.isFinite && j > element.minimumDeterminant { low = mid } else { high = mid }
                }
                safe = min(safe, low * 0.95)
            }
        }
        for element in elements {
            let j = deformationGradient(element, state: state, direction: direction, alpha: safe).determinant
            minJ = min(minJ, j); minVolume = min(minVolume, j * element.restVolume)
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: minJ,
            minimumVolume: minVolume, minimumRodLength: 1,
            finite: minJ.isFinite && minVolume.isFinite, safeStep: safe)
    }

    private func positions(_ element: NXIncompressibleTetrahedron, state: [Float],
                           direction: [Float]? = nil, alpha: Float = 0) -> [SIMD3<Float>] {
        var x = element.previousPositions
        for node in 0..<4 {
            let o = Int(element.velocityOffsets[node])
            var v = SIMD3<Float>(state[o], state[o + 1], state[o + 2])
            if let direction { v += alpha * SIMD3<Float>(direction[o], direction[o + 1], direction[o + 2]) }
            x[node] += timeStepSeconds * v
        }
        return x
    }

    private func deformationGradient(_ element: NXIncompressibleTetrahedron, state: [Float],
                                     direction: [Float]? = nil, alpha: Float = 0) -> NXMatrix3 {
        let x = positions(element, state: state, direction: direction, alpha: alpha)
        return NXMatrix3(x[1] - x[0], x[2] - x[0], x[3] - x[0]).multiplied(by: element.inverseRestMatrix)
    }

    private func directionalGradient(_ element: NXIncompressibleTetrahedron, vector: [Float]) -> NXMatrix3 {
        func dv(_ node: Int) -> SIMD3<Float> {
            let o = Int(element.velocityOffsets[node]); return SIMD3<Float>(vector[o], vector[o + 1], vector[o + 2])
        }
        return NXMatrix3(dv(1) - dv(0), dv(2) - dv(0), dv(3) - dv(0)).multiplied(by: element.inverseRestMatrix)
    }

    private func shapeGradients(_ element: NXIncompressibleTetrahedron) -> [SIMD3<Float>] {
        let invT = element.inverseRestMatrix.transposed
        let g1 = invT.c0, g2 = invT.c1, g3 = invT.c2
        return [-(g1 + g2 + g3), g1, g2, g3]
    }
}
