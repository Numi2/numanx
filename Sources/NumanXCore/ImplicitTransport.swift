import Foundation

public struct NXTransportNode: Codable, Equatable, Sendable {
    public var stateIndex: Int
    public var previousValue: Float
    public var capacity: Float
    public var sourceRate: Float
    public var linearReactionRate: Float

    public init(stateIndex: Int, previousValue: Float, capacity: Float = 1,
                sourceRate: Float = 0, linearReactionRate: Float = 0) throws {
        guard stateIndex >= 0, previousValue.isFinite, capacity.isFinite, capacity > 0,
              sourceRate.isFinite, linearReactionRate.isFinite else {
            throw NXError.invalidConfiguration("transport node")
        }
        self.stateIndex = stateIndex; self.previousValue = previousValue; self.capacity = capacity
        self.sourceRate = sourceRate; self.linearReactionRate = linearReactionRate
    }
}

public struct NXTransportEdge: Codable, Equatable, Sendable {
    public var firstNode: Int
    public var secondNode: Int
    public var conductance: Float

    public init(firstNode: Int, secondNode: Int, conductance: Float) throws {
        guard firstNode >= 0, secondNode >= 0, firstNode != secondNode,
              conductance.isFinite, conductance >= 0 else {
            throw NXError.invalidConfiguration("transport edge")
        }
        self.firstNode = firstNode; self.secondNode = secondNode; self.conductance = conductance
    }
}

/// Backward-Euler conservative scalar transport on a graph. It supports heat, oxygen, dissolved
/// species or other fields that are locally diffusive with linear reaction/source terms. Nonlinear
/// constitutive reactions can be additional contributions to the same state.
public struct NXImplicitTransportContribution: NXPhysicsContribution {
    public let name = "implicit-conservative-transport"
    public let stateDimension: Int
    public let timeStepSeconds: Float
    public let nodes: [NXTransportNode]
    public let edges: [NXTransportEdge]

    public init(stateDimension: Int, timeStepSeconds: Float,
                nodes: [NXTransportNode], edges: [NXTransportEdge]) throws {
        guard stateDimension > 0, timeStepSeconds.isFinite, timeStepSeconds > 0, !nodes.isEmpty else {
            throw NXError.invalidConfiguration("transport contribution")
        }
        guard Set(nodes.map(\.stateIndex)).count == nodes.count,
              nodes.allSatisfy({ $0.stateIndex < stateDimension }),
              edges.allSatisfy({ $0.firstNode < nodes.count && $0.secondNode < nodes.count }) else {
            throw NXError.invalidConfiguration("transport topology")
        }
        self.stateDimension = stateDimension; self.timeStepSeconds = timeStepSeconds
        self.nodes = nodes; self.edges = edges
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("transport residual")
        }
        for node in nodes {
            let x = state[node.stateIndex]
            residual[node.stateIndex] += node.capacity * (x - node.previousValue)
                - timeStepSeconds * (node.sourceRate + node.linearReactionRate * x)
        }
        for edge in edges {
            let a = nodes[edge.firstNode].stateIndex
            let b = nodes[edge.secondNode].stateIndex
            let flux = timeStepSeconds * edge.conductance * (state[a] - state[b])
            residual[a] += flux; residual[b] -= flux
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("transport residual") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension else {
            throw NXError.invalidState("transport Jv")
        }
        for node in nodes {
            product[node.stateIndex] += (node.capacity - timeStepSeconds * node.linearReactionRate) * vector[node.stateIndex]
        }
        for edge in edges {
            let a = nodes[edge.firstNode].stateIndex
            let b = nodes[edge.secondNode].stateIndex
            let flux = timeStepSeconds * edge.conductance * (vector[a] - vector[b])
            product[a] += flux; product[b] -= flux
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("transport Jv") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("transport safe step")
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }
}
