// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script, console2} from "forge-std/Script.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
import {ShieldedCredits} from "../src/ShieldedCredits.sol";
import {RLNSettlement} from "../src/RLNSettlement.sol";

/// @title Deploy — vLLM Inference Blueprint Contracts
/// @notice Deploys InferenceBSM, ShieldedCredits, and RLNSettlement.
///
/// Usage:
///   export PRIVATE_KEY=0x...
///   forge script script/Deploy.s.sol:Deploy --rpc-url $RPC_URL --broadcast --slow
///
/// Environment variables:
///   PRIVATE_KEY          — deployer private key
///   TSUSD_ADDRESS        — tsUSD token address (shielded pool wrapped stablecoin)
///   TANGLE_CORE_ADDRESS  — Tangle core contract address (optional, set post-deploy)
contract Deploy is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        address tsUSD = vm.envOr("TSUSD_ADDRESS", address(0));
        address tangleCore = vm.envOr("TANGLE_CORE_ADDRESS", address(0));

        vm.startBroadcast(deployerKey);

        // 1. Deploy ShieldedCredits
        ShieldedCredits credits = new ShieldedCredits();
        console2.log("ShieldedCredits:", address(credits));

        // 2. Deploy RLNSettlement
        RLNSettlement rln = new RLNSettlement();
        console2.log("RLNSettlement:", address(rln));

        // 3. Deploy InferenceBSM (UUPS proxy)
        if (tsUSD == address(0)) {
            console2.log("WARNING: TSUSD_ADDRESS not set, using address(1) as placeholder");
            tsUSD = address(1);
        }
        InferenceBSM impl = new InferenceBSM();
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(impl),
            abi.encodeCall(InferenceBSM.initialize, (tsUSD))
        );
        InferenceBSM bsm = InferenceBSM(payable(address(proxy)));
        console2.log("InferenceBSM impl:", address(impl));
        console2.log("InferenceBSM proxy:", address(bsm));

        // 4. If Tangle core address provided, initialize BSM
        if (tangleCore != address(0)) {
            bsm.onBlueprintCreated(0, msg.sender, tangleCore);
            console2.log("BSM initialized with Tangle core:", tangleCore);
        }

        // 5. Configure a default model (operators can register against this)
        bsm.configureModel(
            "Qwen/Qwen2-0.5B",  // model name
            4096,                 // max context length
            1,                    // price per input token (tsUSD base units)
            2,                    // price per output token
            2048                  // min GPU VRAM MiB
        );
        console2.log("Default model configured: Qwen/Qwen2-0.5B");

        vm.stopBroadcast();

        // Summary
        console2.log("");
        console2.log("=== Deployment Summary ===");
        console2.log("ShieldedCredits:", address(credits));
        console2.log("RLNSettlement:  ", address(rln));
        console2.log("InferenceBSM:    ", address(bsm), "(proxy)");
        console2.log("tsUSD:          ", tsUSD);
    }
}
