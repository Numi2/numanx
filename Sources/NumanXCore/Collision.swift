import Foundation

public struct NXAABB: Codable, Equatable, Sendable {
    public var minimum: SIMD3<Float>
    public var maximum: SIMD3<Float>

    public init(minimum: SIMD3<Float>, maximum: SIMD3<Float>) throws {
        guard minimum.x.isFinite, minimum.y.isFinite, minimum.z.isFinite,
              maximum.x.isFinite, maximum.y.isFinite, maximum.z.isFinite,
              minimum.x <= maximum.x, minimum.y <= maximum.y, minimum.z <= maximum.z else {
            throw NXError.invalidConfiguration("AABB")
        }
        self.minimum = minimum; self.maximum = maximum
    }

    public func overlaps(_ other: NXAABB) -> Bool {
        minimum.x <= other.maximum.x && maximum.x >= other.minimum.x
            && minimum.y <= other.maximum.y && maximum.y >= other.minimum.y
            && minimum.z <= other.maximum.z && maximum.z >= other.minimum.z
    }

    public static func sweptSphere(from start: SIMD3<Float>, to end: SIMD3<Float>, radius: Float) throws -> NXAABB {
        guard radius.isFinite, radius >= 0 else { throw NXError.invalidConfiguration("sphere radius") }
        let r = SIMD3<Float>(repeating: radius)
        return try NXAABB(minimum: SIMD3<Float>(Swift.min(start.x, end.x), Swift.min(start.y, end.y), Swift.min(start.z, end.z)) - r,
                          maximum: SIMD3<Float>(Swift.max(start.x, end.x), Swift.max(start.y, end.y), Swift.max(start.z, end.z)) + r)
    }
}

public struct NXRigidSphere: Codable, Equatable, Sendable {
    public var identifier: UInt64
    public var collisionGroup: UInt32
    public var collisionMask: UInt32
    public var generalizedVelocityOffset: Int
    public var previousCenter: SIMD3<Float>
    /// World-space vector from body COM to sphere center at the beginning of the step.
    public var leverArm: SIMD3<Float>
    public var radius: Float
    public var frictionCoefficient: Float

    public init(identifier: UInt64, collisionGroup: UInt32 = 1, collisionMask: UInt32 = .max,
                generalizedVelocityOffset: Int, previousCenter: SIMD3<Float>,
                leverArm: SIMD3<Float> = .zero, radius: Float, frictionCoefficient: Float) throws {
        guard identifier > 0, generalizedVelocityOffset >= 0,
              previousCenter.x.isFinite, previousCenter.y.isFinite, previousCenter.z.isFinite,
              leverArm.x.isFinite, leverArm.y.isFinite, leverArm.z.isFinite,
              radius.isFinite, radius > 0, frictionCoefficient.isFinite, frictionCoefficient >= 0 else {
            throw NXError.invalidConfiguration("rigid sphere")
        }
        self.identifier = identifier; self.collisionGroup = collisionGroup; self.collisionMask = collisionMask
        self.generalizedVelocityOffset = generalizedVelocityOffset; self.previousCenter = previousCenter
        self.leverArm = leverArm; self.radius = radius; self.frictionCoefficient = frictionCoefficient
    }
}

public struct NXCollisionPair: Hashable, Codable, Sendable, Comparable {
    public var first: UInt64
    public var second: UInt64
    public init(_ a: UInt64, _ b: UInt64) {
        first = Swift.min(a, b); second = Swift.max(a, b)
    }
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.first == rhs.first ? lhs.second < rhs.second : lhs.first < rhs.first
    }
}

public struct NXSphereContactGeometry: Equatable, Sendable {
    public var pair: NXCollisionPair
    public var normal: SIMD3<Float>
    public var gap: Float
    public var timeOfImpactFraction: Float?
}

/// Deterministic sort-and-sweep reference broadphase. Production Metal uses the same pair ordering
/// after GPU candidate generation so persistent contact history does not depend on scheduling order.
public enum NXSweepAndPruneBroadphase {
    private struct Entry {
        var shape: NXRigidSphere
        var box: NXAABB
    }

    public static func candidates(shapes: [NXRigidSphere], state: [Float],
                                  timeStepSeconds: Float, padding: Float = 0) throws -> [NXCollisionPair] {
        guard timeStepSeconds.isFinite, timeStepSeconds > 0, padding.isFinite, padding >= 0,
              Set(shapes.map(\.identifier)).count == shapes.count else {
            throw NXError.invalidConfiguration("broadphase")
        }
        var entries: [Entry] = []
        entries.reserveCapacity(shapes.count)
        for shape in shapes {
            guard shape.generalizedVelocityOffset <= state.count - 6 else { throw NXError.invalidState("sphere body state") }
            let end = shape.previousCenter + timeStepSeconds * centerVelocity(shape, state: state)
            entries.append(Entry(shape: shape,
                box: try .sweptSphere(from: shape.previousCenter, to: end, radius: shape.radius + padding)))
        }
        entries.sort {
            if $0.box.minimum.x != $1.box.minimum.x { return $0.box.minimum.x < $1.box.minimum.x }
            return $0.shape.identifier < $1.shape.identifier
        }
        var result = Set<NXCollisionPair>()
        for i in entries.indices {
            let a = entries[i]
            var j = i + 1
            while j < entries.count, entries[j].box.minimum.x <= a.box.maximum.x {
                let b = entries[j]
                if a.box.overlaps(b.box), filter(a.shape, b.shape) {
                    result.insert(NXCollisionPair(a.shape.identifier, b.shape.identifier))
                }
                j += 1
            }
        }
        return result.sorted()
    }

    private static func filter(_ a: NXRigidSphere, _ b: NXRigidSphere) -> Bool {
        a.identifier != b.identifier && (a.collisionMask & b.collisionGroup) != 0 && (b.collisionMask & a.collisionGroup) != 0
    }
}

public enum NXSphereSphereCCD {
    /// Exact time of impact for linearly swept sphere centers. Returns fraction in [0,1].
    public static func timeOfImpact(a: NXRigidSphere, b: NXRigidSphere,
                                    state: [Float], timeStepSeconds: Float,
                                    extraSeparation: Float = 0) throws -> Float? {
        guard timeStepSeconds.isFinite, timeStepSeconds > 0, extraSeparation.isFinite, extraSeparation >= 0,
              a.generalizedVelocityOffset <= state.count - 6,
              b.generalizedVelocityOffset <= state.count - 6 else { throw NXError.invalidState("sphere CCD") }
        let p = b.previousCenter - a.previousCenter
        let displacement = timeStepSeconds * (centerVelocity(b, state: state) - centerVelocity(a, state: state))
        let radius = a.radius + b.radius + extraSeparation
        let c = dot(p, p) - radius * radius
        if c <= 0 { return 0 }
        let aa = dot(displacement, displacement)
        if aa <= Float.leastNormalMagnitude { return nil }
        let bb = 2 * dot(p, displacement)
        let discriminant = bb * bb - 4 * aa * c
        if discriminant < 0 { return nil }
        let root = sqrt(max(0, discriminant))
        let q = -0.5 * (bb + (bb >= 0 ? root : -root))
        let t0: Float
        let t1: Float
        if abs(q) <= Float.leastNormalMagnitude {
            t0 = -bb / (2 * aa); t1 = t0
        } else {
            t0 = q / aa; t1 = c / q
        }
        let first = min(t0, t1), second = max(t0, t1)
        if first >= 0 && first <= 1 { return first }
        if second >= 0 && second <= 1 { return second }
        return nil
    }

    public static func geometry(a: NXRigidSphere, b: NXRigidSphere, state: [Float],
                                timeStepSeconds: Float) throws -> NXSphereContactGeometry {
        let ca = a.previousCenter + timeStepSeconds * centerVelocity(a, state: state)
        let cb = b.previousCenter + timeStepSeconds * centerVelocity(b, state: state)
        let delta = cb - ca
        let distance = sqrt(dot(delta, delta))
        let normal: SIMD3<Float>
        if distance > 1e-8 { normal = delta / distance }
        else {
            let seed = a.identifier < b.identifier ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(-1, 0, 0)
            normal = seed
        }
        return NXSphereContactGeometry(pair: NXCollisionPair(a.identifier, b.identifier), normal: normal,
            gap: distance - a.radius - b.radius,
            timeOfImpactFraction: try timeOfImpact(a: a, b: b, state: state, timeStepSeconds: timeStepSeconds))
    }
}

public enum NXRigidSphereContactBuilder {
    /// Creates a three-row contact Jacobian in [linear xyz, angular xyz] generalized-velocity order.
    /// Multiplier ordering is [normal,tangent1,tangent2].
    public static func contactRow(a: NXRigidSphere, b: NXRigidSphere,
                                  state: [Float], multiplierOffset: Int,
                                  timeStepSeconds: Float, stabilizationFraction: Float = 0.2,
                                  compliance: Float = 0, rho: Float = 1) throws -> NXContactRow {
        guard stabilizationFraction.isFinite, stabilizationFraction >= 0, stabilizationFraction <= 1 else {
            throw NXError.invalidConfiguration("contact stabilization")
        }
        let geometry = try NXSphereSphereCCD.geometry(a: a, b: b, state: state, timeStepSeconds: timeStepSeconds)
        let n = geometry.normal
        let t1 = tangent(n)
        let t2 = cross(n, t1)
        let normal = try jacobian(a: a, b: b, direction: n)
        let tangent1 = try jacobian(a: a, b: b, direction: t1)
        let tangent2 = try jacobian(a: a, b: b, direction: t2)
        let mu = sqrt(a.frictionCoefficient * b.frictionCoefficient)
        let stabilization = geometry.gap < 0 ? stabilizationFraction * geometry.gap / timeStepSeconds : 0
        return try NXContactRow(multiplierOffset: multiplierOffset,
            normal: normal, tangent1: tangent1, tangent2: tangent2,
            law: NXCoulombContact(frictionCoefficient: mu, normalCompliance: compliance,
                tangentialCompliance: compliance, stabilizationVelocity: stabilization), rho: rho)
    }

    private static func jacobian(a: NXRigidSphere, b: NXRigidSphere,
                                 direction d: SIMD3<Float>) throws -> [NXJacobianEntry] {
        let raCross = cross(a.leverArm, d)
        let rbCross = cross(b.leverArm, d)
        var result: [NXJacobianEntry] = []
        result.reserveCapacity(12)
        for axis in 0..<3 {
            if d[axis] != 0 {
                result.append(try NXJacobianEntry(index: a.generalizedVelocityOffset + axis, coefficient: -d[axis]))
                result.append(try NXJacobianEntry(index: b.generalizedVelocityOffset + axis, coefficient: d[axis]))
            }
            if raCross[axis] != 0 {
                result.append(try NXJacobianEntry(index: a.generalizedVelocityOffset + 3 + axis, coefficient: -raCross[axis]))
            }
            if rbCross[axis] != 0 {
                result.append(try NXJacobianEntry(index: b.generalizedVelocityOffset + 3 + axis, coefficient: rbCross[axis]))
            }
        }
        // Multiple physical contributions can land on the same generalized coordinate (e.g. zero
        // lever-arm angular term disappears). Combine deterministically before NXContactRow validation.
        var sums: [Int: Float] = [:]
        for entry in result { sums[entry.index, default: 0] += entry.coefficient }
        return try sums.keys.sorted().compactMap { index in
            let value = sums[index]!
            return abs(value) > 1e-12 ? try NXJacobianEntry(index: index, coefficient: value) : nil
        }
    }

    private static func tangent(_ n: SIMD3<Float>) -> SIMD3<Float> {
        let axis = abs(n.x) < 0.57735 ? SIMD3<Float>(1, 0, 0)
            : (abs(n.y) < 0.57735 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(0, 0, 1))
        let t = cross(axis, n)
        let length = sqrt(dot(t, t))
        return t / max(length, 1e-12)
    }
}

private func centerVelocity(_ shape: NXRigidSphere, state: [Float]) -> SIMD3<Float> {
    let o = shape.generalizedVelocityOffset
    let linear = SIMD3<Float>(state[o], state[o + 1], state[o + 2])
    let angular = SIMD3<Float>(state[o + 3], state[o + 4], state[o + 5])
    return linear + cross(angular, shape.leverArm)
}

@inline(__always) private func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x*b.x + a.y*b.y + a.z*b.z
}
@inline(__always) private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
}
