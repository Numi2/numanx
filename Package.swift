// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "NumanX",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "NumanXCore", targets: ["NumanXCore"]),
        .library(name: "NumanXMetal", targets: ["NumanXMetal"]),
        .library(name: "NumanXRecovery", targets: ["NumanXRecovery"]),
        .executable(name: "numanx-recovery", targets: ["NumanXRecoveryCLI"])
    ],
    targets: [
        .target(name: "NumanXCore"),
        .target(name: "NumanXMetal", dependencies: ["NumanXCore"], resources: [.process("Shaders")]),
        .target(name: "NumanXRecovery"),
        .executableTarget(name: "NumanXRecoveryCLI", dependencies: ["NumanXRecovery"]),
        .testTarget(name: "NumanXCoreTests", dependencies: ["NumanXCore"]),
        .testTarget(name: "NumanXRecoveryTests", dependencies: ["NumanXRecovery"])
    ],
    swiftLanguageModes: [.v6]
)
