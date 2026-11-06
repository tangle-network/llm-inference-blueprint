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
        vm.startBroadcast();
        InferenceBSM implementation = new InferenceBSM();
        address manager = address(
            new ERC1967Proxy(address(implementation), abi.encodeCall(InferenceBSM.initialize, (token, registry)))
        );
        console2.log("DEPLOY_INFERENCE_BSM_IMPL=%s", vm.toString(address(implementation)));
        vm.stopBroadcast();
        console2.log("DEPLOY_INFERENCE_BSM_PROXY=%s", vm.toString(manager));
    }
}
