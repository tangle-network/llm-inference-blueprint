// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title IRLNSettlement
/// @notice Rate-Limiting Nullifier settlement for shielded payments.
///
/// @dev Users deposit tokens against an identity commitment. Each epoch, the user
///      generates an RLN proof off-chain; the operator verifies it and later batch-claims.
///      If a user double-signals within an epoch, anyone can submit two Shamir shares
///      to slash the deposit.
interface IRLNSettlement {
    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event Deposited(bytes32 indexed identityCommitment, address indexed token, uint256 amount);
    event BatchClaimed(address indexed operator, uint256 count, uint256 totalAmount);
    event Slashed(bytes32 indexed identityCommitment, address indexed slasher, uint256 amount);
    event Withdrawn(bytes32 indexed identityCommitment, address indexed recipient, uint256 amount);

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error NullifierUsed(bytes32 nullifier);
    error InsufficientDeposit(uint256 available, uint256 requested);
    error InvalidSlash();
    error SlashFailed();

    // ═══════════════════════════════════════════════════════════════════════
    // FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    function deposit(address token, uint256 amount, bytes32 identityCommitment) external;

    function batchClaim(
        address token,
        bytes32[] calldata nullifiers,
        uint256[] calldata amounts,
        address operator
    )
        external;

    function slash(
        bytes32 nullifier,
        uint256 x1,
        uint256 y1,
        uint256 x2,
        uint256 y2,
        bytes32 identityCommitment
    )
        external;

    function withdraw(bytes32 identityCommitment, uint256 amount, bytes calldata proof) external;
}
