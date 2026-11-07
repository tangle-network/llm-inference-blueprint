// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("PAYMENT_TOKEN");
        // The model's operator config is the one source of its served model, prices and VRAM floor.
        string memory config = vm.readFile(vm.envOr("MODEL_CONFIG", string("../deploy/config.example.json")));
        vm.startBroadcast();
        InferenceBSM implementation = new InferenceBSM();
        address manager = address(
            new ERC1967Proxy(address(implementation), abi.encodeCall(InferenceBSM.initialize, (token, registry)))
        );
        InferenceBSM(payable(manager)).configureModel(
            vm.parseJsonString(config, ".vllm.model"),
            uint32(vm.parseJsonUint(config, ".vllm.max_model_len")),
            uint64(vm.parseJsonUint(config, ".vllm.price_per_input_token")),
            uint64(vm.parseJsonUint(config, ".vllm.price_per_output_token")),
            uint32(vm.parseJsonUint(config, ".gpu.min_vram_mib"))
        );
        vm.stopBroadcast();
        console2.log("DEPLOY_MANAGER=%s", vm.toString(manager));
    }
}
