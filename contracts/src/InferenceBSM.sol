// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import { UUPSUpgradeable } from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

/// @title InferenceBSM
/// @notice Blueprint Service Manager for vLLM inference services.
/// @dev Operators register with GPU capabilities. Services only accept the configured
///      payment token (e.g. USDC wrapped via VAnchor) for anonymous billing.
contract InferenceBSM is Initializable, UUPSUpgradeable, BlueprintServiceManagerBase {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

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
        uint64 pricePerInputToken; // in payment token base units
        uint64 pricePerOutputToken; // in payment token base units
        uint32 minGpuVramMib; // Minimum VRAM required to serve this model
        bool enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice The accepted payment token (e.g. USDC wrapped via VAnchor).
    address public paymentToken;

    /// @notice Minimum operator stake (in TNT)
    uint256 public constant MIN_OPERATOR_STAKE = 100 ether;

    /// @notice operator => capabilities
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// @notice model name hash => ModelConfig
    mapping(bytes32 => ModelConfig) public modelConfigs;

    /// @notice Set of registered operators
    EnumerableSet.AddressSet private _operators;

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @notice Initialize the contract (called once via proxy)
    /// @param _paymentToken The ERC20 token accepted for payment (e.g. USDC wrapped via VAnchor)
    function initialize(address _paymentToken) external initializer {
        __UUPSUpgradeable_init();
        paymentToken = _paymentToken;
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
    // OPERATOR LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Operator registers with GPU capabilities.
    /// @param registrationInputs abi.encode(string model, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
    function onRegister(address operator, bytes calldata registrationInputs) external payable override onlyFromTangle {
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

    function onUnregister(address operator) external override onlyFromTangle {
        operatorCaps[operator].active = false;
        _operators.remove(operator);
    }

    function onUpdatePreferences(address operator, bytes calldata newPreferences) external payable override onlyFromTangle {
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
    ) external payable override onlyFromTangle {
        // Only accept the configured payment token
        if (paymentAsset != paymentToken && paymentAsset != address(0)) {
            revert InvalidPaymentAsset(paymentAsset);
        }
    }

    function onServiceInitialized(
        uint64,
        uint64,
        uint64 serviceId,
        address,
        address[] calldata,
        uint64
    ) external override onlyFromTangle {
        // Restrict payment to the configured payment token for this service
        _permitAsset(serviceId, paymentToken);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DYNAMIC MEMBERSHIP
    // ═══════════════════════════════════════════════════════════════════════

    function canJoin(uint64, address operator) external view override returns (bool) {
        return operatorCaps[operator].active;
    }

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
    ) external payable override onlyFromTangle {
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
    ) external payable override onlyFromTangle {
        if (!operatorCaps[operator].active) revert OperatorNotRegistered(operator);

        (, uint32 promptTokens, uint32 completionTokens) = abi.decode(outputs, (string, uint32, uint32));

        uint32 totalTokens = promptTokens + completionTokens;
        emit InferenceResultSubmitted(serviceId, jobCallId, totalTokens);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES
    // ═══════════════════════════════════════════════════════════════════════

    function queryIsPaymentAssetAllowed(uint64 serviceId, address asset) external view override returns (bool) {
        if (asset == address(0)) return true;
        address[] memory permitted = _getPermittedAssets(serviceId);
        if (permitted.length == 0) return asset == paymentToken;
        for (uint256 i; i < permitted.length; ++i) {
            if (permitted[i] == asset) return true;
        }
        return false;
    }

    function getAggregationThreshold(uint64, uint8) external pure override returns (uint16, uint8) {
        return (0, 0);
    }

    function getMinOperatorStake() external pure override returns (bool, uint256) {
        return (false, MIN_OPERATOR_STAKE);
    }

    function getHeartbeatInterval(uint64) external pure override returns (bool, uint64) {
        return (false, 100); // ~100 blocks heartbeat for liveness
    }

    function getExitConfig(uint64) external pure override returns (bool, uint64, uint64, bool) {
        // 1 hour min commitment, 1 hour exit queue, force exit allowed
        return (false, 3600, 3600, true);
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

    // ═══════════════════════════════════════════════════════════════════════
    // UPGRADES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Only the blueprint owner can authorize upgrades
    function _authorizeUpgrade(address) internal override onlyBlueprintOwner {}

    receive() external payable override {}
}
