// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
import {BlueprintServiceManagerBase} from "tnt-core/BlueprintServiceManagerBase.sol";

contract InferenceBSMTest is Test {
    InferenceBSM public bsm;
    address public paymentToken;

    address public tangleCore = address(0xC0DE);
    address public owner = address(0xBEEF);
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        paymentToken = address(0xD00D);

        // Deploy implementation + proxy
        InferenceBSM impl = new InferenceBSM();
        bytes memory initData = abi.encodeCall(InferenceBSM.initialize, (address(paymentToken), tangleCore));
        vm.prank(owner);
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        bsm = InferenceBSM(payable(address(proxy)));

        // Initialize blueprint
        vm.prank(tangleCore);
        bsm.onBlueprintCreated(1, owner, tangleCore);

        // Configure a model
        vm.prank(owner);
        bsm.configureModel(
            "meta-llama/Llama-3.1-8B-Instruct",
            8192, // max context
            1, // 1 base unit per input token
            2, // 2 base units per output token
            16_000 // 16GB VRAM minimum
        );
    }

    // ─── Initialization ──────────────────────────────────────────────────

    // ─── Model Configuration ─────────────────────────────────────────────

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test-model", 4096, 1, 1, 8000);
    }

    function test_disableModel() public {
        vm.prank(owner);
        bsm.disableModel("meta-llama/Llama-3.1-8B-Instruct");

        InferenceBSM.ModelConfig memory mc = bsm.getModelConfig("meta-llama/Llama-3.1-8B-Instruct");
        assertFalse(mc.enabled);
    }

    // ─── Operator Registration ───────────────────────────────────────────

    function test_registerOperator() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(2), uint32(48_000), "NVIDIA A100", "https://op1.example.com"
        );

        vm.prank(tangleCore);
        bsm.onRegister(operator1, regData);

        assertTrue(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 1);
    }

    function test_registerOperator_unsupportedModel() public {
        bytes memory regData =
            abi.encode("nonexistent-model", uint32(1), uint32(16_000), "NVIDIA A100", "https://op1.example.com");

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.ModelNotSupported.selector, "nonexistent-model"));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_insufficientVram() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct",
            uint32(1),
            uint32(8_000), // Only 8GB, need 16GB
            "NVIDIA RTX 3060",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InsufficientGpuCapability.selector, 16_000, 8_000));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_onlyTangle() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(1), uint32(16_000), "NVIDIA A100", "https://op1.example.com"
        );

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // ─── Unregistration ──────────────────────────────────────────────────

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 0);
    }

    // ─── Service Request ─────────────────────────────────────────────────

    function test_onRequest_invalidPaymentAsset() public {
        address[] memory ops = new address[](0);
        address wrongToken = address(0xDEAD);

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InvalidPaymentAsset.selector, wrongToken));
        bsm.onRequest(1, user, ops, "", 3600, wrongToken, 1000);
    }

    // ─── Job Lifecycle ───────────────────────────────────────────────────

    function test_onJobResult_unregisteredOperator() public {
        bytes memory inputs = abi.encode("test", uint32(10), uint64(7000));
        bytes memory outputs = abi.encode("result", uint32(2), uint32(3));

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.OperatorNotRegistered.selector, operator1));
        bsm.onJobResult(1, 0, 1, operator1, inputs, outputs);
    }

    // ─── Configuration Queries ───────────────────────────────────────────

    function test_paymentAssetAllowed_paymentToken() public {
        // Initialize a service to set up permitted assets
        address[] memory callers = new address[](0);
        vm.prank(tangleCore);
        bsm.onServiceInitialized(1, 1, 1, owner, callers, 3600);

        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(paymentToken)));
        assertFalse(bsm.queryIsPaymentAssetAllowed(1, address(0xDEAD)));
        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(0))); // native always allowed
    }

    // ─── Dynamic Membership ─────────────────────────────────────────────

    function test_canJoin_activeOperator() public {
        _registerOperator(operator1);
        assertTrue(bsm.canJoin(1, operator1));
    }

    // ─── Operator Pricing Query ────────────────────────────────────────

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 inputPrice, uint64 outputPrice, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(inputPrice, 1);
        assertEq(outputPrice, 2);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://operator.example.com")));
    }

    function test_getOperatorPricing_unregistered() public {
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.OperatorNotRegistered.selector, operator1));
        bsm.getOperatorPricing(operator1);
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    function _registerOperator(address op) internal {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(2), uint32(48_000), "NVIDIA A100", "https://operator.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}
