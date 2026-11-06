// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {console2} from "forge-std/Script.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

contract Deploy is RegisterBlueprint {
    function _deployManager() internal override returns (address) {
        return _deployManagerFor(vm.envAddress("PAYMENT_TOKEN"), vm.envAddress("TANGLE_CORE"));
    }

    function _deployManagerFor(address token, address registry) internal returns (address) {
        InferenceBSM impl = new InferenceBSM();
        console2.log("DEPLOY_INFERENCE_BSM_IMPL=%s", vm.toString(address(impl)));
        return address(new ERC1967Proxy(address(impl), abi.encodeCall(InferenceBSM.initialize, (token, registry))));
    }

    function _guardNetwork(address tangleAddr) internal pure override {
        require(tangleAddr != 0xff137B9C879c47C28ce389e84501925438AB4cdA, "Tempo: use script/DeployTempo.s.sol");
    }

    function _report(address manager, uint64 blueprintId) internal override {
        console2.log("DEPLOY_INFERENCE_BSM_PROXY=%s", vm.toString(manager));
        console2.log("DEPLOY_INFERENCE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }
}
