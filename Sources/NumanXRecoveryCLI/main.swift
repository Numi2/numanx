import Foundation
import NumanXRecovery

@main
struct NumanXRecoveryCLI {
    static func main() async {
        do { try await run(Array(CommandLine.arguments.dropFirst())) }
        catch {
            FileHandle.standardError.write(Data("numanx-recovery: \(error)\n".utf8))
            exit(1)
        }
    }
    static func run(_ args: [String]) async throws {
        guard let command = args.first else { help(); return }
        switch command {
        case "help", "--help": help()
        case "status":
            guard args.count == 1 else { throw CLIError.usage }
            try emit(["implementation": "durable-native-image-store-and-recovery-coordinator",
                      "solver": "not-implemented-in-this-repository",
                      "compiled-or-tested-in-development-session": "false",
                      "physical-hardware-control": "unsupported",
                      "automatic-decision": "never-inferred-from-presence-of-a-prepare-file"])
        case "inspect":
            guard args.count == 3, let transaction = UInt64(args[2], radix: 16), transaction > 0 else { throw CLIError.usage }
            let store = try FileNumanXPreparedStateStore(directoryURL: URL(fileURLWithPath: args[1]))
            do {
                let image = try await store.loadPrepared(transactionFingerprint: transaction)
                let decision = try await store.decision(transactionFingerprint: transaction)
                try emit(Inspection(manifest: image.manifest, decision: decision,
                    verifiedAuthoritativeBytes: image.authoritativeState.count,
                    verifiedGPUBytes: image.gpuBufferImage.count))
                try await store.close()
            } catch {
                try? await store.close()
                throw error
            }
        default: throw CLIError.usage
        }
    }
    struct Inspection: Encodable {
        var manifest: NumanXPreparedStateManifest
        var decision: NumanXRecoveryDecision
        var verifiedAuthoritativeBytes: Int
        var verifiedGPUBytes: Int
    }
    static func emit<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        FileHandle.standardOutput.write(try encoder.encode(value))
        FileHandle.standardOutput.write(Data([10]))
    }
    static func help() {
        print("""
        numanx-recovery status
        numanx-recovery inspect <existing-store-directory> <transaction-fingerprint-hex>

        Inspect verifies persisted bytes and reports the stored decision. It never creates a commit
        decision, restores a solver, or publishes a generation. It requires the exclusive store lock.
        """)
    }
    enum CLIError: Error { case usage }
}
