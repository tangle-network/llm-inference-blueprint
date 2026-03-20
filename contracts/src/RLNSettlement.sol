// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { IERC20 } from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import { SafeERC20 } from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import { ReentrancyGuard } from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

import { IRLNSettlement } from "./IRLNSettlement.sol";

/// @title RLNSettlement
/// @notice Minimal settlement contract for RLN-based shielded payments.
contract RLNSettlement is IRLNSettlement, ReentrancyGuard {
    using SafeERC20 for IERC20;

    uint256 internal constant FIELD_PRIME =
        21_888_242_871_839_275_222_246_405_745_257_275_088_548_364_400_416_034_343_698_204_186_575_808_495_617;

    struct DepositInfo {
        address token;
        uint256 balance;
        address depositor;
    }

    mapping(bytes32 => DepositInfo) public deposits;
    mapping(bytes32 => bool) public usedNullifiers;
    mapping(address => bool) public authorizedOperators;
    address public owner;

    struct PendingSlash {
        bytes32 identityCommitment;
        address slasher;
        uint256 amount;
        uint256 claimableAt;
    }

    mapping(bytes32 => PendingSlash) public pendingSlashes;
    uint256 public constant SLASH_DELAY = 1 days;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    function registerOperator(address op) external onlyOwner {
        authorizedOperators[op] = true;
    }

    function removeOperator(address op) external onlyOwner {
        authorizedOperators[op] = false;
    }

    function deposit(address token, uint256 amount, bytes32 identityCommitment) external nonReentrant {
        if (amount == 0) revert InsufficientDeposit(0, 0);

        DepositInfo storage info = deposits[identityCommitment];

        if (info.depositor == address(0)) {
            info.token = token;
            info.depositor = msg.sender;
        } else {
            if (info.token != token) revert InsufficientDeposit(0, amount);
        }

        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        info.balance += amount;

        emit Deposited(identityCommitment, token, amount);
    }

    function batchClaim(
        address token,
        bytes32[] calldata nullifiers,
        uint256[] calldata amounts,
        address operator
    )
        external
        nonReentrant
    {
        require(authorizedOperators[msg.sender], "not authorized operator");

        uint256 len = nullifiers.length;
        require(len == amounts.length, "length mismatch");
        require(len > 0, "empty batch");

        uint256 totalAmount;

        for (uint256 i; i < len; ++i) {
            bytes32 nf = nullifiers[i];
            if (usedNullifiers[nf]) revert NullifierUsed(nf);
            usedNullifiers[nf] = true;
            totalAmount += amounts[i];
        }

        if (totalAmount > 0) {
            IERC20(token).safeTransfer(operator, totalAmount);
        }

        emit BatchClaimed(operator, len, totalAmount);
    }

    function slash(
        bytes32, /* nullifier */
        uint256 x1,
        uint256 y1,
        uint256 x2,
        uint256 y2,
        bytes32 identityCommitment
    )
        external
        nonReentrant
    {
        if (x1 == x2) revert InvalidSlash();

        uint256 dy = addmod(y2, FIELD_PRIME - y1, FIELD_PRIME);
        uint256 dx = addmod(x2, FIELD_PRIME - x1, FIELD_PRIME);
        uint256 dxInv = _modInverse(dx, FIELD_PRIME);
        uint256 slope = mulmod(dy, dxInv, FIELD_PRIME);
        uint256 secret = addmod(y1, FIELD_PRIME - mulmod(x1, slope, FIELD_PRIME), FIELD_PRIME);

        DepositInfo storage info = deposits[identityCommitment];
        if (info.balance == 0) revert SlashFailed();

        bytes32 slashId = keccak256(abi.encode(identityCommitment, x1, y1, x2, y2));
        require(pendingSlashes[slashId].amount == 0, "slash already pending");

        uint256 slashAmount = info.balance;
        info.balance = 0;

        pendingSlashes[slashId] = PendingSlash({
            identityCommitment: identityCommitment,
            slasher: msg.sender,
            amount: slashAmount,
            claimableAt: block.timestamp + SLASH_DELAY
        });

        emit Slashed(identityCommitment, msg.sender, slashAmount);
    }

    function finalizeSlash(bytes32 slashId) external nonReentrant {
        PendingSlash storage ps = pendingSlashes[slashId];
        require(ps.amount > 0, "no pending slash");
        require(block.timestamp >= ps.claimableAt, "slash not claimable yet");

        uint256 amount = ps.amount;
        address slasher = ps.slasher;
        bytes32 ic = ps.identityCommitment;
        ps.amount = 0;

        DepositInfo storage info = deposits[ic];
        IERC20(info.token).safeTransfer(slasher, amount);
    }

    function withdraw(
        bytes32 identityCommitment,
        uint256 amount,
        bytes calldata /* proof */
    )
        external
        nonReentrant
    {
        DepositInfo storage info = deposits[identityCommitment];
        require(msg.sender == info.depositor, "not depositor");
        if (amount > info.balance) revert InsufficientDeposit(info.balance, amount);

        info.balance -= amount;
        IERC20(info.token).safeTransfer(msg.sender, amount);

        emit Withdrawn(identityCommitment, msg.sender, amount);
    }

    function getDeposit(bytes32 identityCommitment) external view returns (address token, uint256 balance) {
        DepositInfo storage info = deposits[identityCommitment];
        return (info.token, info.balance);
    }

    function _modInverse(uint256 a, uint256 p) internal view returns (uint256) {
        if (a == 0) revert InvalidSlash();
        return _modExp(a, p - 2, p);
    }

    function _modExp(uint256 b, uint256 e, uint256 m) internal view returns (uint256 result) {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, 0x20)
            mstore(add(ptr, 0x20), 0x20)
            mstore(add(ptr, 0x40), 0x20)
            mstore(add(ptr, 0x60), b)
            mstore(add(ptr, 0x80), e)
            mstore(add(ptr, 0xa0), m)
            if iszero(staticcall(gas(), 0x05, ptr, 0xc0, ptr, 0x20)) { revert(0, 0) }
            result := mload(ptr)
        }
    }
}
