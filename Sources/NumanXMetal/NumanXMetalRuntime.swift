import Foundation
@preconcurrency import Metal
import NumanXCore

@available(macOS 26.0, *)
public struct NXMetalRuntimeIdentity: Codable, Equatable, Sendable {
    public var deviceRegistryID: UInt64
    public var deviceName: String
    public var solverProgramFingerprint: UInt64
    public var maximumThreadgroupWidth: Int
    public var recommendedMaxWorkingSetSize: UInt64
}

@available(macOS 26.0, *)
public final class NXMetalRuntime: @unchecked Sendable {
    public let device: any MTLDevice
    public let identity: NXMetalRuntimeIdentity
    public let commandQueue: any MTL4CommandQueue
    public let residencySet: any MTLResidencySet
    private let commandAllocator: any MTL4CommandAllocator
    private let commandBuffer: any MTL4CommandBuffer
    private let lock = NSLock()
    private var transactionOpen = false
    private var commandBufferInUse = false

    public init(device: any MTLDevice, solverProgramFingerprint: UInt64,
                initialResidencyCapacity: Int = 64) throws {
        guard solverProgramFingerprint > 0, initialResidencyCapacity > 0,
              let queue = device.makeMTL4CommandQueue(),
              let allocator = device.makeCommandAllocator(),
              let buffer = device.makeCommandBuffer() else {
            throw NXError.invalidConfiguration("Metal 4 runtime")
        }
        let descriptor = MTLResidencySetDescriptor()
        descriptor.label = "NumanX authoritative solver residency"
        descriptor.initialCapacity = initialResidencyCapacity
        let residency = try device.makeResidencySet(descriptor: descriptor)
        residency.commit(); residency.requestResidency()
        self.device = device; commandQueue = queue; commandAllocator = allocator
        commandBuffer = buffer; residencySet = residency
        identity = NXMetalRuntimeIdentity(deviceRegistryID: device.registryID,
            deviceName: device.name, solverProgramFingerprint: solverProgramFingerprint,
            maximumThreadgroupWidth: 256,
            recommendedMaxWorkingSetSize: UInt64(device.recommendedMaxWorkingSetSize))
    }

    deinit { residencySet.endResidency() }

    public func addPersistentAllocation(_ allocation: any MTLBuffer) throws {
        lock.lock(); defer { lock.unlock() }
        guard !transactionOpen, allocation.device.registryID == device.registryID else {
            throw NXError.invalidState("residency mutation during transaction or foreign device")
        }
        residencySet.addAllocation(allocation)
        residencySet.commit()
    }

    public func beginTransaction() throws {
        lock.lock(); defer { lock.unlock() }
        guard !transactionOpen, !commandBufferInUse else { throw NXError.invalidState("Metal transaction already open") }
        transactionOpen = true
    }

    public func finishTransaction() throws {
        lock.lock(); defer { lock.unlock() }
        guard transactionOpen, !commandBufferInUse else { throw NXError.invalidState("Metal command still encoded/in flight") }
        transactionOpen = false
    }

    /// Encodes one pass into the reusable Metal 4 command buffer. The caller owns end/commit and
    /// completion feedback because publication must remain part of the outer transaction timeline.
    /// `releaseCommandBufferAfterOwnerCompletion` must be called only after terminal feedback.
    public func beginAuthoritativeCommandBuffer() throws -> any MTL4CommandBuffer {
        lock.lock(); defer { lock.unlock() }
        guard transactionOpen, !commandBufferInUse else { throw NXError.invalidState("Metal command lifecycle") }
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        commandBuffer.useResidencySet(residencySet)
        commandBuffer.label = "NumanX authoritative shadow solve"
        commandBufferInUse = true
        return commandBuffer
    }

    public func releaseCommandBufferAfterOwnerCompletion() throws {
        lock.lock(); defer { lock.unlock() }
        guard transactionOpen, commandBufferInUse else { throw NXError.invalidState("no command buffer to release") }
        commandBufferInUse = false
    }
}

/// Fixed dispatch budget keeps large-batch training deterministic. Adaptive scientific execution
/// can raise the budget between roots, but never mutate it while an authoritative transaction runs.
public struct NXMetalDispatchBudget: Codable, Equatable, Sendable {
    public var newtonIterations: Int
    public var krylovIterationsPerNewton: Int
    public var patchPassesPerKrylov: Int
    public var conePassesPerNewton: Int
    public var reductionPassesPerKrylov: Int

    public init(profile: NXExecutionProfile, patchPassesPerKrylov: Int = 2,
                conePassesPerNewton: Int = 2, reductionPassesPerKrylov: Int = 4) throws {
        guard patchPassesPerKrylov > 0, conePassesPerNewton > 0, reductionPassesPerKrylov > 0 else {
            throw NXError.invalidConfiguration("dispatch budget")
        }
        newtonIterations = profile.maximumNewtonIterations
        krylovIterationsPerNewton = profile.maximumKrylovIterations
        self.patchPassesPerKrylov = patchPassesPerKrylov
        self.conePassesPerNewton = conePassesPerNewton
        self.reductionPassesPerKrylov = reductionPassesPerKrylov
    }
    public var maximumKernelDispatches: Int {
        newtonIterations * (krylovIterationsPerNewton * (patchPassesPerKrylov + reductionPassesPerKrylov) + conePassesPerNewton)
    }
}
