// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { RLNSettlement } from "shielded-payment-gateway/src/shielded/RLNSettlement.sol";
import { IRLNSettlement } from "shielded-payment-gateway/src/shielded/IRLNSettlement.sol";

/// @title MockERC20 for RLN billing tests
contract MockToken {
    string public name = "MockUSD";
    string public symbol = "MUSD";
    uint8 public decimals = 18;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

/// @title RLNBillingE2ETest
/// @notice Proves the full RLN billing flow for a vLLM operator:
///         1. User deposits tokens against an identity commitment
///         2. User generates RLN proofs off-chain (verified by operator)
///         3. Operator batch-claims payments on-chain with unique nullifiers
///         4. Double-signal detection via Shamir shares enables slashing
///         5. Unauthorized operators cannot claim
contract RLNBillingE2ETest is Test {
    RLNSettlement public settlement;
    MockToken public token;

    address public operator = makeAddr("vllm-operator");
    address public depositor = makeAddr("user");
    address public attacker = makeAddr("attacker");

    uint256 internal constant FIELD_PRIME =
        21_888_242_871_839_275_222_246_405_745_257_275_088_548_364_400_416_034_343_698_204_186_575_808_495_617;

    uint256 internal identitySecret = 42;
    bytes32 internal identityCommitment;

    uint256 constant DEPOSIT_AMOUNT = 100 ether;
    uint256 constant INFERENCE_COST = 0.01 ether;

    function setUp() public {
        settlement = new RLNSettlement();
        token = new MockToken();
        identityCommitment = keccak256(abi.encodePacked(identitySecret));

        // Register the operator
        settlement.registerOperator(operator);

        // Fund and deposit
        token.mint(depositor, DEPOSIT_AMOUNT);
        vm.prank(depositor);
        token.approve(address(settlement), type(uint256).max);
        vm.prank(depositor);
        settlement.deposit(address(token), DEPOSIT_AMOUNT, identityCommitment);
    }

    /// @notice Simulate batch claiming 50 inference requests.
    function test_batchClaimInferenceRequests() public {
        uint256 batchSize = 50;

        bytes32[] memory nullifiers = new bytes32[](batchSize);
        uint256[] memory amounts = new uint256[](batchSize);

        for (uint256 i = 0; i < batchSize; i++) {
            nullifiers[i] = keccak256(abi.encodePacked("inference-nullifier", i));
            amounts[i] = INFERENCE_COST;
        }

        vm.prank(operator);
        settlement.batchClaim(address(token), nullifiers, amounts, operator);

        assertEq(token.balanceOf(operator), INFERENCE_COST * batchSize);

        // All nullifiers are marked as used
        for (uint256 i = 0; i < batchSize; i++) {
            assertTrue(settlement.usedNullifiers(nullifiers[i]));
        }
    }

    /// @notice Multiple batches with unique nullifiers succeed.
    function test_multipleBatches() public {
        // Batch 1
        bytes32[] memory nf1 = new bytes32[](2);
        nf1[0] = keccak256("batch1-nf0");
        nf1[1] = keccak256("batch1-nf1");
        uint256[] memory am1 = new uint256[](2);
        am1[0] = 1 ether;
        am1[1] = 2 ether;

        vm.prank(operator);
        settlement.batchClaim(address(token), nf1, am1, operator);
        assertEq(token.balanceOf(operator), 3 ether);

        // Batch 2 with different nullifiers
        bytes32[] memory nf2 = new bytes32[](1);
        nf2[0] = keccak256("batch2-nf0");
        uint256[] memory am2 = new uint256[](1);
        am2[0] = 4 ether;

        vm.prank(operator);
        settlement.batchClaim(address(token), nf2, am2, operator);
        assertEq(token.balanceOf(operator), 7 ether);
    }

    /// @notice Reusing a nullifier from a previous batch reverts.
    function test_reusedNullifierReverts() public {
        bytes32[] memory nf = new bytes32[](1);
        nf[0] = keccak256("unique-nf");
        uint256[] memory am = new uint256[](1);
        am[0] = 1 ether;

        vm.prank(operator);
        settlement.batchClaim(address(token), nf, am, operator);

        // Same nullifier again
        vm.expectRevert(abi.encodeWithSelector(IRLNSettlement.NullifierUsed.selector, nf[0]));
        vm.prank(operator);
        settlement.batchClaim(address(token), nf, am, operator);
    }

    /// @notice Unauthorized operator cannot batch-claim.
    function test_unauthorizedOperatorReverts() public {
        bytes32[] memory nf = new bytes32[](1);
        nf[0] = keccak256("nf");
        uint256[] memory am = new uint256[](1);
        am[0] = 1 ether;

        vm.prank(attacker);
        vm.expectRevert("not authorized operator");
        settlement.batchClaim(address(token), nf, am, attacker);
    }

    /// @notice Removed operator cannot batch-claim.
    function test_removedOperatorReverts() public {
        settlement.removeOperator(operator);

        bytes32[] memory nf = new bytes32[](1);
        nf[0] = keccak256("nf");
        uint256[] memory am = new uint256[](1);
        am[0] = 1 ether;

        vm.prank(operator);
        vm.expectRevert("not authorized operator");
        settlement.batchClaim(address(token), nf, am, operator);
    }

    /// @notice Slash a double-signaler with two Shamir shares.
    function test_slashDoubleSignal() public {
        address slasher = makeAddr("slasher");

        // Construct two Shamir shares: y = 42 + 7*x
        uint256 x1 = 1;
        uint256 y1 = addmod(identitySecret, mulmod(7, x1, FIELD_PRIME), FIELD_PRIME);
        uint256 x2 = 3;
        uint256 y2 = addmod(identitySecret, mulmod(7, x2, FIELD_PRIME), FIELD_PRIME);

        bytes32 nullifier = keccak256("double-signal");

        vm.prank(slasher);
        settlement.slash(nullifier, x1, y1, x2, y2, identityCommitment);

        // Balance locked, not yet paid
        (, uint256 bal,) = settlement.getDeposit(identityCommitment);
        assertEq(bal, 0);
        assertEq(token.balanceOf(slasher), 0);

        // Finalize after delay
        bytes32 slashId = keccak256(abi.encode(identityCommitment, x1, y1, x2, y2));
        vm.warp(block.timestamp + settlement.SLASH_DELAY() + 1);
        settlement.finalizeSlash(slashId);

        assertEq(token.balanceOf(slasher), DEPOSIT_AMOUNT);
    }

    /// @notice Cannot slash with same x-coordinate (not two distinct shares).
    function test_slashSameXReverts() public {
        vm.expectRevert(IRLNSettlement.InvalidSlash.selector);
        settlement.slash(keccak256("nf"), 1, 10, 1, 20, identityCommitment);
    }

    /// @notice Cannot finalize slash before delay expires.
    function test_slashBeforeDelayReverts() public {
        uint256 x1 = 1;
        uint256 y1 = addmod(identitySecret, mulmod(7, x1, FIELD_PRIME), FIELD_PRIME);
        uint256 x2 = 3;
        uint256 y2 = addmod(identitySecret, mulmod(7, x2, FIELD_PRIME), FIELD_PRIME);

        settlement.slash(keccak256("nf"), x1, y1, x2, y2, identityCommitment);

        bytes32 slashId = keccak256(abi.encode(identityCommitment, x1, y1, x2, y2));
        vm.expectRevert("slash not claimable yet");
        settlement.finalizeSlash(slashId);
    }

    /// @notice Fuzz: random batch sizes and amounts maintain invariants.
    function testFuzz_batchClaimInvariants(uint8 batchSize, uint256 perCost) public {
        batchSize = uint8(bound(batchSize, 1, 50));
        perCost = bound(perCost, 1, DEPOSIT_AMOUNT / uint256(batchSize));

        bytes32[] memory nf = new bytes32[](batchSize);
        uint256[] memory am = new uint256[](batchSize);
        uint256 totalCost;

        for (uint256 i = 0; i < batchSize; i++) {
            nf[i] = keccak256(abi.encodePacked("fuzz-nf", i, batchSize, perCost));
            am[i] = perCost;
            totalCost += perCost;
        }

        vm.prank(operator);
        settlement.batchClaim(address(token), nf, am, operator);

        assertEq(token.balanceOf(operator), totalCost);
    }
}
