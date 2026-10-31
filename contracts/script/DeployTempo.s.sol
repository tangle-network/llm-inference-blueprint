// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import { Script, console2 } from "forge-std/Script.sol";
import { ERC1967Proxy } from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { InferenceBSM } from "../src/InferenceBSM.sol";
interface IERC20Decimals {
    function decimals() external view returns (uint8);
}
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
    function setBlueprintSettlementAsset(uint64 blueprintId, address asset) external;
    function setJobEventRates(uint64 blueprintId, uint8[] calldata jobIndexes, uint256[] calldata rates) external;
    function getBlueprintSettlementAsset(uint64 blueprintId) external view returns (address);
    function getJobEventRate(uint64 blueprintId, uint8 jobIndex) external view returns (uint256);
}
/// @notice Registers a new UUPS LLM manager on Tempo Moderato and pins a flat
///         floor (default 0.001 token). Blueprint 10 cannot take pathUSD.
contract DeployTempo is Script {
    address constant TANGLE = 0xff137B9C879c47C28ce389e84501925438AB4cdA;
    address constant PATH_USD = 0x20C0000000000000000000000000000000000000;
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address tangleAddr = vm.envOr("TANGLE_CORE", TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", PATH_USD);
        uint8 decimals = IERC20Decimals(paymentToken).decimals();
        uint256 floor = vm.envOr("EVENT_RATE", _defaultFloor(decimals));
        require(floor > 0 && floor <= _maxFloor(decimals), "EVENT_RATE out of range for settlement decimals");
        ITangle tangle = ITangle(tangleAddr);
        vm.startBroadcast(deployerKey);
        InferenceBSM impl = new InferenceBSM();
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), abi.encodeCall(InferenceBSM.initialize, (paymentToken)));
        InferenceBSM bsm = InferenceBSM(payable(address(proxy)));
        uint64 blueprintId = tangle.createBlueprint(_definition(address(bsm), floor));
        tangle.setBlueprintSettlementAsset(blueprintId, paymentToken);
        uint8[] memory jobs = new uint8[](1);
        uint256[] memory rates = new uint256[](1);
        jobs[0] = 0;
        rates[0] = floor;
        tangle.setJobEventRates(blueprintId, jobs, rates);
        vm.stopBroadcast();
        require(tangle.getBlueprintSettlementAsset(blueprintId) == paymentToken, "settlement asset was not pinned");
        require(tangle.getJobEventRate(blueprintId, 0) == floor, "inference floor was not written");
        console2.log("DEPLOY_INFERENCE_BSM_IMPL=%s", vm.toString(address(impl)));
        console2.log("DEPLOY_INFERENCE_BSM_PROXY=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_INFERENCE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
        console2.log("floor (token units):", floor);
    }
    function _defaultFloor(uint8 decimals) internal pure returns (uint256) {
        if (decimals > 18) return 0;
        uint256 floor = (10 ** uint256(decimals)) / 1000;
        return floor == 0 ? 1 : floor;
    }
    function _maxFloor(uint8 decimals) internal pure returns (uint256) {
        if (decimals > 18) return 0;
        return 1_000_000 * (10 ** uint256(decimals));
    }
    function _definition(address manager, uint256 floor) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "tangle:llm-inference";
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0,
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: floor
        });
        def.metadata = Types.BlueprintMetadata({
            name: "LLM Inference Blueprint",
            description: "Routes a listed model to a provider. Other models stay local. The floor is the on-chain minimum.",
            author: "Tangle",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/llm-inference-blueprint",
            logo: "",
            website: "https://tangle.network",
            license: "MIT",
            profilingData: ""
        });
        def.jobs = new Types.JobDefinition[](1);
        def.jobs[0] = Types.JobDefinition({
            name: "inference", description: "", metadataUri: "", paramsSchema: "", resultSchema: ""
        });
        def.registrationSchema = "";
        def.requestSchema = "";
        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "bin",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(Types.BlueprintFetcherKind.None, "bin", "bin"),
            testing: Types.TestingSource("bin", "bin", "."),
            binaries: bins
        });
        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }
}
