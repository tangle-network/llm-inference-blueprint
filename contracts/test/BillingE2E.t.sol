// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { ShieldedCredits } from "../src/ShieldedCredits.sol";
import { IShieldedCredits } from "../src/IShieldedCredits.sol";
import { MessageHashUtils } from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/// @title MockERC20 for testing
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

/// @title BillingE2ETest
/// @notice Proves the full billing flow between a vLLM operator and ShieldedCredits:
///         1. User funds credits (simulating gateway flow)
///         2. User signs SpendAuth (EIP-712)
///         3. Operator calls authorizeSpend on-chain
///         4. Operator claims payment after serving inference
///         5. User reclaims expired auths
contract BillingE2ETest is Test {
    ShieldedCredits public credits;
    MockToken public token;

    // Ephemeral user keys
    uint256 internal userPrivKey = 0xdead;
    address internal userPubKey;
    bytes32 internal salt = keccak256("billing-e2e");
    bytes32 internal commitment;

    // Operator
    address public operator = makeAddr("vllm-operator");

    // Funding
    address public funder = makeAddr("gateway");
    uint256 constant CREDIT_AMOUNT = 100 ether;
    uint256 constant INFERENCE_COST = 0.01 ether; // ~$0.01 per inference

    function setUp() public {
        credits = new ShieldedCredits();
        token = new MockToken();

        userPubKey = vm.addr(userPrivKey);
        commitment = keccak256(abi.encodePacked(userPubKey, salt));

        // Fund the "gateway" with tokens and approve credits contract
        token.mint(funder, CREDIT_AMOUNT);
        vm.prank(funder);
        token.approve(address(credits), type(uint256).max);

        // Fund credits (simulates ShieldedGateway.shieldedFundCredits)
        vm.prank(funder);
        credits.fundCredits(address(token), CREDIT_AMOUNT, commitment, userPubKey);
    }

    /// @notice Simulate 100 inference requests, each costing 0.01 tokens.
    ///         This proves the billing integration handles high-frequency payments.
    function test_100InferenceRequests() public {
        uint256 numRequests = 100;

        for (uint256 i = 0; i < numRequests; i++) {
            // User signs SpendAuth off-chain
            IShieldedCredits.SpendAuth memory auth = _signSpend(
                uint64(1),      // serviceId
                uint8(0),       // jobIndex (inference)
                INFERENCE_COST,
                i               // nonce
            );

            // Operator submits authorizeSpend on-chain
            bytes32 authHash = credits.authorizeSpend(auth);

            // Operator claims after serving inference
            vm.prank(operator);
            credits.claimPayment(authHash, operator);
        }

        // Verify: operator received all payments
        assertEq(token.balanceOf(operator), INFERENCE_COST * numRequests);

        // Verify: credit balance decreased correctly
        IShieldedCredits.CreditAccountView memory acct = credits.getAccount(commitment);
        assertEq(acct.balance, CREDIT_AMOUNT - (INFERENCE_COST * numRequests));
        assertEq(acct.nonce, numRequests);
    }

    /// @notice Simulate the real operator billing flow:
    ///         1. User sends SpendAuth with max amount
    ///         2. Operator serves inference (actual cost < max)
    ///         3. Operator claims full pre-auth (overpays)
    ///         4. User can't reclaim before expiry
    ///         5. For the next request, new nonce works
    function test_operatorBillingFlow() public {
        uint256 maxCost = 0.05 ether;

        // Request 1: authorize max cost, operator claims full amount
        IShieldedCredits.SpendAuth memory auth1 = _signSpend(1, 0, maxCost, 0);
        bytes32 authHash1 = credits.authorizeSpend(auth1);

        vm.prank(operator);
        credits.claimPayment(authHash1, operator);
        assertEq(token.balanceOf(operator), maxCost);

        // Request 2: same flow, next nonce
        IShieldedCredits.SpendAuth memory auth2 = _signSpend(1, 0, maxCost, 1);
        bytes32 authHash2 = credits.authorizeSpend(auth2);

        vm.prank(operator);
        credits.claimPayment(authHash2, operator);
        assertEq(token.balanceOf(operator), maxCost * 2);

        // Verify balance
        IShieldedCredits.CreditAccountView memory acct = credits.getAccount(commitment);
        assertEq(acct.balance, CREDIT_AMOUNT - (maxCost * 2));
    }

    /// @notice Test expired authorization reclaim flow
    function test_expiredAuthReclaim() public {
        uint64 shortExpiry = uint64(block.timestamp + 60); // 60 seconds

        IShieldedCredits.SpendAuth memory auth = _signSpendWithExpiry(1, 0, 1 ether, 0, shortExpiry);
        bytes32 authHash = credits.authorizeSpend(auth);

        // Balance deducted
        assertEq(credits.getAccount(commitment).balance, CREDIT_AMOUNT - 1 ether);

        // Operator doesn't claim in time
        vm.warp(block.timestamp + 61);

        // Operator can't claim after expiry
        vm.prank(operator);
        vm.expectRevert();
        credits.claimPayment(authHash, operator);

        // Anyone can reclaim expired auth
        credits.reclaimExpiredAuth(authHash, commitment);

        // Balance restored
        assertEq(credits.getAccount(commitment).balance, CREDIT_AMOUNT);
    }

    /// @notice Test that wrong operator can't claim
    function test_wrongOperatorCannotClaim() public {
        IShieldedCredits.SpendAuth memory auth = _signSpend(1, 0, 1 ether, 0);
        bytes32 authHash = credits.authorizeSpend(auth);

        address attacker = makeAddr("attacker");
        vm.prank(attacker);
        vm.expectRevert();
        credits.claimPayment(authHash, attacker);
    }

    /// @notice Fuzz: random amounts and nonces maintain balance invariant
    function testFuzz_balanceInvariant(uint256 amount1, uint256 amount2) public {
        amount1 = bound(amount1, 1, CREDIT_AMOUNT / 2);
        amount2 = bound(amount2, 1, CREDIT_AMOUNT / 2);

        IShieldedCredits.SpendAuth memory auth1 = _signSpend(1, 0, amount1, 0);
        credits.authorizeSpend(auth1);

        IShieldedCredits.SpendAuth memory auth2 = _signSpend(1, 0, amount2, 1);
        credits.authorizeSpend(auth2);

        IShieldedCredits.CreditAccountView memory acct = credits.getAccount(commitment);
        assertEq(acct.balance + acct.totalSpent, acct.totalFunded);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    function _signSpend(
        uint64 serviceId,
        uint8 jobIndex,
        uint256 amount,
        uint256 nonce
    ) internal view returns (IShieldedCredits.SpendAuth memory) {
        return _signSpendWithExpiry(serviceId, jobIndex, amount, nonce, uint64(block.timestamp) + 3600);
    }

    function _signSpendWithExpiry(
        uint64 serviceId,
        uint8 jobIndex,
        uint256 amount,
        uint256 nonce,
        uint64 expiry
    ) internal view returns (IShieldedCredits.SpendAuth memory) {
        bytes32 structHash = keccak256(
            abi.encode(
                credits.SPEND_TYPEHASH(), commitment, serviceId, jobIndex, amount, operator, nonce, expiry
            )
        );
        bytes32 digest = MessageHashUtils.toTypedDataHash(credits.DOMAIN_SEPARATOR(), structHash);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(userPrivKey, digest);

        return IShieldedCredits.SpendAuth({
            commitment: commitment,
            serviceId: serviceId,
            jobIndex: jobIndex,
            amount: amount,
            operator: operator,
            nonce: nonce,
            expiry: expiry,
            signature: abi.encodePacked(r, s, v)
        });
    }
}
