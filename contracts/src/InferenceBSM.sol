// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

// In production, import from tnt-core's npm/soldeer package:
// import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";
// import { IShieldedCredits } from "tnt-core/shielded/IShieldedCredits.sol";
//
// For now we define minimal interfaces inline so the contract compiles standalone.

/// @dev Minimal IBlueprintServiceManager interface (matches tnt-core)
interface IBlueprintServiceManager {
    function onBlueprintCreated(uint64 blueprintId, address owner, address tangleCore) external;
    function onRegister(address operator, bytes calldata registrationInputs) external payable;
    function onUnregister(address operator) external;
    function onUpdatePreferences(address operator, bytes calldata newPreferences) external payable;
    function getHeartbeatInterval(uint64 serviceId) external view returns (bool useDefault, uint64 interval);
    function getHeartbeatThreshold(uint64 serviceId) external view returns (bool useDefault, uint8 threshold);
    function getSlashingWindow(uint64 serviceId) external view returns (bool useDefault, uint64 window);

    function getExitConfig(uint64 serviceId)
        external
        view
        returns (bool useDefault, uint64 minCommitmentDuration, uint64 exitQueueDuration, bool forceExitAllowed);

    function getNonPaymentTerminationPolicy(uint64 serviceId)
        external
        view
        returns (bool useDefault, uint64 graceIntervals);

    function onRequest(
        uint64 requestId,
        address requester,
        address[] calldata operators,
        bytes calldata requestInputs,
        uint64 ttl,
        address paymentAsset,
        uint256 paymentAmount
    ) external payable;

    function onApprove(address operator, uint64 requestId, uint8 stakingPercent) external payable;
    function onReject(address operator, uint64 requestId) external;

    function onServiceInitialized(
        uint64 blueprintId,
        uint64 requestId,
        uint64 serviceId,
        address owner,
        address[] calldata permittedCallers,
        uint64 ttl
    ) external;

    function onServiceTermination(uint64 serviceId, address owner) external;
    function canJoin(uint64 serviceId, address operator) external view returns (bool);
    function onOperatorJoined(uint64 serviceId, address operator, uint16 exposureBps) external;
    function canLeave(uint64 serviceId, address operator) external view returns (bool);
    function onOperatorLeft(uint64 serviceId, address operator) external;
    function onExitScheduled(uint64 serviceId, address operator, uint64 executeAfter) external;
    function onExitCanceled(uint64 serviceId, address operator) external;
    function onJobCall(uint64 serviceId, uint8 job, uint64 jobCallId, bytes calldata inputs) external payable;

    function onJobResult(
        uint64 serviceId,
        uint8 job,
        uint64 jobCallId,
        address operator,
        bytes calldata inputs,
        bytes calldata outputs
    ) external payable;

    function onUnappliedSlash(uint64 serviceId, bytes calldata offender, uint8 slashPercent) external;
    function onSlash(uint64 serviceId, bytes calldata offender, uint8 slashPercent) external;
    function querySlashingOrigin(uint64 serviceId) external view returns (address);
    function queryDisputeOrigin(uint64 serviceId) external view returns (address);
    function queryDeveloperPaymentAddress(uint64 serviceId) external view returns (address payable);
    function queryIsPaymentAssetAllowed(uint64 serviceId, address asset) external view returns (bool);
    function getRequiredResultCount(uint64 serviceId, uint8 jobIndex) external view returns (uint32);
    function requiresAggregation(uint64 serviceId, uint8 jobIndex) external view returns (bool);
    function getAggregationThreshold(uint64 serviceId, uint8 jobIndex) external view returns (uint16, uint8);

    function onAggregatedResult(
        uint64 serviceId,
        uint8 job,
        uint64 jobCallId,
        bytes calldata output,
        uint256 signerBitmap,
        uint256[2] calldata aggregatedSignature,
        uint256[4] calldata aggregatedPubkey
    ) external;

    function getMinOperatorStake() external view returns (bool useDefault, uint256 minStake);
}

/// @title InferenceBSM
/// @notice Blueprint Service Manager for vLLM inference services.
/// @dev Operators register with GPU capabilities. Services only accept tsUSD payment
///      (the ShieldedCredits wrapped token) for anonymous billing.
contract InferenceBSM is IBlueprintServiceManager {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error OnlyTangleAllowed(address caller, address tangle);
    error OnlyBlueprintOwnerAllowed(address caller, address owner);
    error AlreadyInitialized();
    error InvalidPaymentAsset(address asset);
    error InsufficientGpuCapability(uint32 required, uint32 provided);
    error ModelNotSupported(string model);
    error OperatorNotRegistered(address operator);

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event OperatorRegistered(address indexed operator, string model, uint32 gpuCount, uint32 totalVramMib);
    event ModelConfigured(string model, uint32 maxContextLen, uint64 pricePerInputToken, uint64 pricePerOutputToken);
    event InferenceJobSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, uint32 maxTokens);
    event InferenceResultSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, uint32 totalTokens);

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice GPU capabilities reported by operator at registration
    struct OperatorCapabilities {
        string model;
        uint32 gpuCount;
        uint32 totalVramMib;
        string gpuModel;
        string endpoint; // Operator's HTTP endpoint URL
        bool active;
    }

    /// @notice Model pricing and metadata
    struct ModelConfig {
        uint32 maxContextLen;
        uint64 pricePerInputToken; // in tsUSD base units
        uint64 pricePerOutputToken; // in tsUSD base units
        uint32 minGpuVramMib; // Minimum VRAM required to serve this model
        bool enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    address public tangleCore;
    uint64 public blueprintId;
    address public blueprintOwner;

    /// @notice The only accepted payment token (tsUSD — shielded pool wrapped token)
    address public immutable tsUSD;

    /// @notice Minimum operator stake (in TNT)
    uint256 public constant MIN_OPERATOR_STAKE = 100 ether;

    /// @notice operator => capabilities
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// @notice model name hash => ModelConfig
    mapping(bytes32 => ModelConfig) public modelConfigs;

    /// @notice Set of registered operators
    EnumerableSet.AddressSet private _operators;

    /// @notice Permitted payment assets per service
    mapping(uint64 => EnumerableSet.AddressSet) private _permittedPaymentAssets;

    // ═══════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════

    modifier onlyFromTangle() {
        _onlyFromTangle();
        _;
    }

    modifier onlyBlueprintOwner() {
        _onlyBlueprintOwner();
        _;
    }

    function _onlyFromTangle() internal view {
        if (msg.sender != tangleCore) revert OnlyTangleAllowed(msg.sender, tangleCore);
    }

    function _onlyBlueprintOwner() internal view {
        if (msg.sender != blueprintOwner) revert OnlyBlueprintOwnerAllowed(msg.sender, blueprintOwner);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════

    /// @param _tsUSD The wrapped stablecoin accepted for payment
    constructor(address _tsUSD) {
        tsUSD = _tsUSD;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Configure a model's pricing and requirements
    function configureModel(
        string calldata model,
        uint32 maxContextLen,
        uint64 pricePerInputToken,
        uint64 pricePerOutputToken,
        uint32 minGpuVramMib
    ) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key] = ModelConfig({
            maxContextLen: maxContextLen,
            pricePerInputToken: pricePerInputToken,
            pricePerOutputToken: pricePerOutputToken,
            minGpuVramMib: minGpuVramMib,
            enabled: true
        });

        emit ModelConfigured(model, maxContextLen, pricePerInputToken, pricePerOutputToken);
    }

    /// @notice Disable a model
    function disableModel(string calldata model) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key].enabled = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BLUEPRINT LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    function onBlueprintCreated(uint64 _blueprintId, address owner, address _tangleCore) external {
        if (tangleCore != address(0)) revert AlreadyInitialized();
        blueprintId = _blueprintId;
        blueprintOwner = owner;
        tangleCore = _tangleCore;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPERATOR LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Operator registers with GPU capabilities.
    /// @param registrationInputs abi.encode(string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
    function onRegister(address operator, bytes calldata registrationInputs) external payable onlyFromTangle {
        (
            string memory model,
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory gpuModel,
            string memory endpoint
        ) = abi.decode(registrationInputs, (string, uint32, uint32, string, string));

        // Validate model is configured
        bytes32 modelKey = keccak256(bytes(model));
        ModelConfig storage mc = modelConfigs[modelKey];
        if (!mc.enabled) revert ModelNotSupported(model);

        // Validate GPU capabilities meet model requirements
        if (totalVramMib < mc.minGpuVramMib) {
            revert InsufficientGpuCapability(mc.minGpuVramMib, totalVramMib);
        }

        operatorCaps[operator] = OperatorCapabilities({
            model: model,
            gpuCount: gpuCount,
            totalVramMib: totalVramMib,
            gpuModel: gpuModel,
            endpoint: endpoint,
            active: true
        });

        _operators.add(operator);

        emit OperatorRegistered(operator, model, gpuCount, totalVramMib);
    }

    function onUnregister(address operator) external onlyFromTangle {
        operatorCaps[operator].active = false;
        _operators.remove(operator);
    }

    function onUpdatePreferences(address operator, bytes calldata newPreferences) external payable onlyFromTangle {
        // Allow updating endpoint
        string memory newEndpoint = abi.decode(newPreferences, (string));
        operatorCaps[operator].endpoint = newEndpoint;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SERVICE LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    function onRequest(
        uint64,
        address,
        address[] calldata,
        bytes calldata,
        uint64,
        address paymentAsset,
        uint256
    ) external payable onlyFromTangle {
        // Only accept tsUSD for payment (enables anonymous ShieldedCredits billing)
        if (paymentAsset != tsUSD && paymentAsset != address(0)) {
            revert InvalidPaymentAsset(paymentAsset);
        }
    }

    function onApprove(address, uint64, uint8) external payable onlyFromTangle {}
    function onReject(address, uint64) external onlyFromTangle {}

    function onServiceInitialized(
        uint64,
        uint64,
        uint64 serviceId,
        address,
        address[] calldata,
        uint64
    ) external onlyFromTangle {
        // Restrict payment to tsUSD for this service
        _permittedPaymentAssets[serviceId].add(tsUSD);
    }

    function onServiceTermination(uint64, address) external onlyFromTangle {}

    // ═══════════════════════════════════════════════════════════════════════
    // DYNAMIC MEMBERSHIP
    // ═══════════════════════════════════════════════════════════════════════

    function canJoin(uint64, address operator) external view returns (bool) {
        return operatorCaps[operator].active;
    }

    function onOperatorJoined(uint64, address, uint16) external onlyFromTangle {}

    function canLeave(uint64, address) external pure returns (bool) {
        return true;
    }

    function onOperatorLeft(uint64, address) external onlyFromTangle {}
    function onExitScheduled(uint64, address, uint64) external onlyFromTangle {}
    function onExitCanceled(uint64, address) external onlyFromTangle {}

    // ═══════════════════════════════════════════════════════════════════════
    // JOB LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Validate an inference job submission
    /// @dev inputs = abi.encode(string prompt, uint32 maxTokens, uint64 temperature)
    function onJobCall(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        bytes calldata inputs
    ) external payable onlyFromTangle {
        (, uint32 maxTokens,) = abi.decode(inputs, (string, uint32, uint64));

        emit InferenceJobSubmitted(serviceId, jobCallId, maxTokens);
    }

    /// @notice Validate an inference job result
    /// @dev outputs = abi.encode(string text, uint32 promptTokens, uint32 completionTokens)
    function onJobResult(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        address operator,
        bytes calldata,
        bytes calldata outputs
    ) external payable onlyFromTangle {
        if (!operatorCaps[operator].active) revert OperatorNotRegistered(operator);

        (, uint32 promptTokens, uint32 completionTokens) = abi.decode(outputs, (string, uint32, uint32));

        uint32 totalTokens = promptTokens + completionTokens;
        emit InferenceResultSubmitted(serviceId, jobCallId, totalTokens);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SLASHING
    // ═══════════════════════════════════════════════════════════════════════

    function onUnappliedSlash(uint64, bytes calldata, uint8) external onlyFromTangle {}
    function onSlash(uint64, bytes calldata, uint8) external onlyFromTangle {}

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES
    // ═══════════════════════════════════════════════════════════════════════

    function querySlashingOrigin(uint64) external view returns (address) {
        return address(this);
    }

    function queryDisputeOrigin(uint64) external view returns (address) {
        return address(this);
    }

    function queryDeveloperPaymentAddress(uint64) external view returns (address payable) {
        return payable(blueprintOwner);
    }

    function queryIsPaymentAssetAllowed(uint64 serviceId, address asset) external view returns (bool) {
        if (asset == address(0)) return true;
        if (_permittedPaymentAssets[serviceId].length() == 0) return asset == tsUSD;
        return _permittedPaymentAssets[serviceId].contains(asset);
    }

    function getRequiredResultCount(uint64, uint8) external pure returns (uint32) {
        return 1; // Single operator result is sufficient for inference
    }

    function requiresAggregation(uint64, uint8) external pure returns (bool) {
        return false; // No BLS aggregation for inference
    }

    function getAggregationThreshold(uint64, uint8) external pure returns (uint16, uint8) {
        return (0, 0);
    }

    function onAggregatedResult(uint64, uint8, uint64, bytes calldata, uint256, uint256[2] calldata, uint256[4] calldata)
        external
        onlyFromTangle
    {}

    function getMinOperatorStake() external pure returns (bool, uint256) {
        return (false, MIN_OPERATOR_STAKE);
    }

    function getHeartbeatInterval(uint64) external pure returns (bool, uint64) {
        return (false, 100); // ~100 blocks heartbeat for liveness
    }

    function getHeartbeatThreshold(uint64) external pure returns (bool, uint8) {
        return (true, 0);
    }

    function getSlashingWindow(uint64) external pure returns (bool, uint64) {
        return (true, 0);
    }

    function getExitConfig(uint64) external pure returns (bool, uint64, uint64, bool) {
        // 1 hour min commitment, 1 hour exit queue, force exit allowed
        return (false, 3600, 3600, true);
    }

    function getNonPaymentTerminationPolicy(uint64) external pure returns (bool, uint64) {
        return (true, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Get all registered operators
    function getOperators() external view returns (address[] memory) {
        return _operators.values();
    }

    /// @notice Get operator count
    function getOperatorCount() external view returns (uint256) {
        return _operators.length();
    }

    /// @notice Get model config by name
    function getModelConfig(string calldata model) external view returns (ModelConfig memory) {
        return modelConfigs[keccak256(bytes(model))];
    }

    /// @notice Check if an operator is registered and active
    function isOperatorActive(address operator) external view returns (bool) {
        return operatorCaps[operator].active;
    }

    /// @notice Get operator pricing for a given operator address.
    /// @dev Returns the model's per-token prices and the operator's endpoint.
    ///      Reverts if the operator is not registered or inactive.
    function getOperatorPricing(address operator)
        external
        view
        returns (uint64 pricePerInputToken, uint64 pricePerOutputToken, string memory endpoint)
    {
        OperatorCapabilities storage caps = operatorCaps[operator];
        if (!caps.active) revert OperatorNotRegistered(operator);

        bytes32 modelKey = keccak256(bytes(caps.model));
        ModelConfig storage mc = modelConfigs[modelKey];

        return (mc.pricePerInputToken, mc.pricePerOutputToken, caps.endpoint);
    }

    receive() external payable {}
}
