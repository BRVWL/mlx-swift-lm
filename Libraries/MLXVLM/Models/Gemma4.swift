// Copyright © 2025 Apple Inc.
//
// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Internal text config decoded from "text_config" key in the VLM JSON.
/// Re-implements Gemma4TextConfiguration to avoid MLXLLM import.
struct G4TextConfig: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let headDim: Int
    let globalHeadDim: Int?
    let rmsNormEps: Float
    let vocabularySize: Int
    let numKeyValueHeads: Int
    let numGlobalKeyValueHeads: Int?
    let numKvSharedLayers: Int?
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float?
    let layerTypes: [String]?
    let ropeParameters: [String: [String: StringOrNumber]]?
    let partialRotaryFactor: Float?
    let attentionKEqV: Bool?
    let useDoubleWideMlp: Bool?
    let hiddenSizePerLayerInput: Int?
    let vocabSizePerLayerInput: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case partialRotaryFactor = "partial_rotary_factor"
        case attentionKEqV = "attention_k_eq_v"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        numKvSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKvSharedLayers)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        ropeParameters = try c.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV)
        useDoubleWideMlp = try c.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp)
        hiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
        vocabSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
    }

    var effectiveLayerTypes: [String] {
        if let layerTypes { return layerTypes }
        let pattern =
            Array(repeating: "sliding_attention", count: slidingWindowPattern - 1) + [
                "full_attention"
            ]
        return (0 ..< numHiddenLayers).map { pattern[$0 % pattern.count] }
    }

    var numCaches: Int { numHiddenLayers - (numKvSharedLayers ?? 0) }
    var hasPerLayerInput: Bool { (hiddenSizePerLayerInput ?? 0) > 0 }
}

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let patchSize: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let defaultOutputLength: Int
    public let ropeParameters: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case defaultOutputLength = "default_output_length"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
    }

    var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 100.0 }
    var maxPatches: Int { defaultOutputLength * poolingKernelSize * poolingKernelSize }
}

public struct Gemma4Configuration: Codable, Sendable {
    public let modelType: String
    let textConfig: G4TextConfig
    public let visionConfig: Gemma4VisionConfiguration
    public let imageTokenId: Int
    public let boiTokenId: Int
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        textConfig = try c.decode(G4TextConfig.self, forKey: .textConfig)
        visionConfig = try c.decode(Gemma4VisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
    }
}

// MARK: - Helpers

/// One-hot encoding: indices [...] → [..., numClasses] float32.
private func oneHotF32(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    let expanded = expandedDimensions(indices, axis: -1)  // [..., 1]
    let range = MLXArray(Int32(0) ..< Int32(numClasses))  // [numClasses]
    return (expanded .== range).asType(.float32)
}

/// Rotate half: [-x2, x1] applied along the last axis.
private func rotateHalfG4(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// 2D RoPE for vision encoder. inputs: [B, L, N, H], positions: [B, L, 2].
private func applyMultidimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float = 100.0
) -> MLXArray {
    let headDim = inputs.dim(-1)
    let ndim = 2
    let channelsPerDim = 2 * (headDim / (2 * ndim))  // 32 for headDim=64
    let halfPerDim = channelsPerDim / 2  // 16

    var parts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * channelsPerDim) ..< ((d + 1) * channelsPerDim)]  // [B,L,N,32]

        let freqExp =
            (2.0 / Float(channelsPerDim))
            * MLXArray(Int32(0) ..< Int32(halfPerDim)).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExp)  // [16]

        let posD = positions[.ellipsis, d ..< (d + 1)].asType(.float32)  // [B, L, 1]
        let sinusoidInp = posD / timescale  // [B, L, 16]
        let cosD = concatenated([cos(sinusoidInp), cos(sinusoidInp)], axis: -1)  // [B, L, 32]
        let sinD = concatenated([sin(sinusoidInp), sin(sinusoidInp)], axis: -1)  // [B, L, 32]

        // Expand to [B, L, 1, 32] for broadcast over numHeads dimension
        let cosDExp = expandedDimensions(cosD, axis: -2).asType(inputs.dtype)
        let sinDExp = expandedDimensions(sinD, axis: -2).asType(inputs.dtype)

        let yPart = xPart * cosDExp + rotateHalfG4(xPart) * sinDExp
        parts.append(yPart)
    }
    return concatenated(parts, axis: -1)
}

/// Functional masked scatter using cumsum (no mutable MLXArray assignment).
/// inputTensor and source must be compatible (same embed dim in last axis).
private func maskedScatterG4(_ inputTensor: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray
{
    let shape = inputTensor.shape
    let maskFlat = mask.flattened().asType(.int32)  // [N]
    let indices = MLX.cumsum(maskFlat, axis: 0) - 1  // [N]
    let sourceFlat = source.flattened()  // [M]
    let sourceSize = sourceFlat.shape[0]
    let clampedIndices = clip(indices, min: Int32(0), max: Int32(sourceSize - 1))
    let aligned = take(sourceFlat, clampedIndices)  // [N]
    return MLX.where(maskFlat.asType(.bool), aligned, inputTensor.flattened()).reshaped(shape)
}

// MARK: - Vision Norms

/// VisionRMSNorm with learnable scale. Uses full float32 for numerical stability.
private class G4VisionRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xF = x.asType(.float32)
        let variance = mean(xF * xF, axis: -1, keepDims: true)
        let normed = xF * rsqrt(variance + eps)
        return (normed * weight.asType(.float32)).asType(x.dtype)
    }
}

/// VisionRMSNorm without scale. Uses full float32 for numerical stability.
private class G4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xF = x.asType(.float32)
        let variance = mean(xF * xF, axis: -1, keepDims: true)
        return (xF * rsqrt(variance + eps)).asType(x.dtype)
    }
}

/// RMSNorm where weight applied directly (no +1 offset). Used for vision transformer block norms.
private class G4RMSNormZeroShift: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// RMSNorm without scale. Uses MLXFast (used for text MultimodalEmbedder norm).
private class G4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

// MARK: - ClippableLinear

/// Wraps Linear with key "linear". Weight path: module.q_proj.linear.weight.
/// Since use_clipped_linears=False for E4B, no clip buffers exist.
private class G4ClippableLinear: Module {
    @ModuleInfo var linear: Linear

    init(inputSize: Int, outputSize: Int) {
        _linear.wrappedValue = Linear(inputSize, outputSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}

// MARK: - Vision Attention

private class G4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: G4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: G4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: G4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: G4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: G4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: G4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: G4VisionRMSNormNoScale

    init(config: Gemma4VisionConfiguration) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        ropeBaseFrequency = config.ropeTheta

        let h = config.hiddenSize
        _qProj.wrappedValue = G4ClippableLinear(inputSize: h, outputSize: numHeads * headDim)
        _kProj.wrappedValue = G4ClippableLinear(inputSize: h, outputSize: numKVHeads * headDim)
        _vProj.wrappedValue = G4ClippableLinear(inputSize: h, outputSize: numKVHeads * headDim)
        _oProj.wrappedValue = G4ClippableLinear(inputSize: numHeads * headDim, outputSize: h)
        _qNorm.wrappedValue = G4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = G4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = G4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)
        v = vNorm(v)

        // Apply 2D RoPE (inputs are [B, L, N, H], positions [B, L, 2])
        q = applyMultidimensionalRoPE(q, positions: positions, baseFrequency: ropeBaseFrequency)
        k = applyMultidimensionalRoPE(k, positions: positions, baseFrequency: ropeBaseFrequency)

        // Transpose to [B, H, L, D]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Vision MLP

private class G4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: G4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: G4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: G4ClippableLinear

    init(config: Gemma4VisionConfiguration) {
        _gateProj.wrappedValue = G4ClippableLinear(
            inputSize: config.hiddenSize, outputSize: config.intermediateSize)
        _upProj.wrappedValue = G4ClippableLinear(
            inputSize: config.hiddenSize, outputSize: config.intermediateSize)
        _downProj.wrappedValue = G4ClippableLinear(
            inputSize: config.intermediateSize, outputSize: config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Vision Transformer Block

private class G4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: G4VisionAttention
    @ModuleInfo var mlp: G4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: G4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: G4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: G4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: G4RMSNormZeroShift

    init(config: Gemma4VisionConfiguration) {
        _selfAttn.wrappedValue = G4VisionAttention(config: config)
        _mlp.wrappedValue = G4VisionMLP(config: config)
        _inputLayernorm.wrappedValue = G4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = G4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = G4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = G4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let attnOut = postAttentionLayernorm(
            selfAttn(inputLayernorm(x), positions: positions, mask: mask))
        let h = x + attnOut
        let mlpOut = postFeedforwardLayernorm(mlp(preFeedforwardLayernorm(h)))
        return h + mlpOut
    }
}

// MARK: - Vision Patch Embedder

private class G4VisionPatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    let patchSize: Int
    let positionEmbeddingSize: Int

    init(config: Gemma4VisionConfiguration) {
        patchSize = config.patchSize
        positionEmbeddingSize = config.positionEmbeddingSize
        _inputProj.wrappedValue = Linear(
            3 * config.patchSize * config.patchSize, config.hiddenSize, bias: false)
        _positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, config.positionEmbeddingSize, config.hiddenSize,
        ])
        super.init()
    }

    /// pixel_values: [B, C, H, W] (channel-first). Returns [B, numPatches, hiddenSize].
    func callAsFunction(
        _ pixelValues: MLXArray,
        patchPositions: MLXArray,  // [B, numPatches, 2] int32
        paddingPositions: MLXArray  // [B, numPatches] bool
    ) -> MLXArray {
        let (B, C, H, W) = (
            pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3)
        )
        let p = patchSize
        let pH = H / p
        let pW = W / p

        // Patchify: [B,C,H,W] → [B, pH*pW, C*p*p]
        var patches = pixelValues.reshaped(B, C, pH, p, pW, p)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)  // [B, pH, pW, p, p, C]
        patches = patches.reshaped(B, pH * pW, C * p * p)
        patches = 2.0 * (patches - 0.5)
        let hiddenStates = inputProj(patches.asType(inputProj.weight.dtype))  // [B, numPatches, hiddenSize]

        // Position embeddings via one-hot lookup into positionEmbeddingTable [2, pos_size, hidden]
        let oh = oneHotF32(patchPositions, numClasses: positionEmbeddingSize)  // [B, numPatches, 2, pos_size]
        let ohT = oh.transposed(0, 2, 1, 3).asType(positionEmbeddingTable.dtype)  // [B, 2, numPatches, pos_size]
        var posEmb = matmul(ohT, positionEmbeddingTable).sum(axis: 1)  // [B, numPatches, hiddenSize]

        // Zero out padding positions
        let padMaskExp = expandedDimensions(paddingPositions, axis: -1).asType(posEmb.dtype)
        posEmb = MLX.where(padMaskExp .!= 0, MLXArray(Float(0)), posEmb)

        return hiddenStates + posEmb
    }
}

// MARK: - Vision Pooler

private class G4VisionPooler {
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        defaultOutputLength = config.defaultOutputLength
        rootHiddenSize = Float(config.hiddenSize).squareRoot()
    }

    func avgPoolByPositions(
        _ x: MLXArray,
        patchPositions: MLXArray,
        length: Int
    ) -> (MLXArray, MLXArray) {
        let inputSeqLen = x.shape[1]
        let k = Int(Double(inputSeqLen / length).squareRoot())
        let kSquared = k * k

        // Clamp positions to >= 0 (padding positions have -1, clamp to 0)
        let clamped = MLX.maximum(patchPositions, MLXArray(Int32(0)))  // [B, L, 2] int32

        // Extract x and y positions
        let xPos = clamped[.ellipsis, 0 ..< 1].squeezed(axes: [-1])  // [B, L]
        let yPos = clamped[.ellipsis, 1...].squeezed(axes: [-1])  // [B, L]

        // max x position + 1 = number of x-columns
        let maxX = xPos.max(axis: -1, keepDims: true) + Int32(1)  // [B, 1]

        // Kernel bin indices
        let kernelX = xPos / Int32(k)  // [B, L]
        let kernelY = yPos / Int32(k)  // [B, L]
        let stride = maxX / Int32(k)  // [B, 1]
        let flatKernelIdxs = kernelX + stride * kernelY  // [B, L]

        // One-hot weights: [B, L, length], normalized by k^2
        let weights = oneHotF32(flatKernelIdxs, numClasses: length) / Float(kSquared)  // [B, L, length]

        // Pool: [B, length, L] @ [B, L, D] → [B, length, D]
        let output = matmul(weights.transposed(0, 2, 1), x.asType(.float32)).asType(x.dtype)

        // Mask: True = valid output position (at least one input patch mapped there)
        let poolMask = logicalNot(all(weights .== Float(0), axis: 1))  // [B, length]

        return (output, poolMask)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        paddingPositions: MLXArray,
        outputLength: Int? = nil
    ) -> (MLXArray, MLXArray) {
        // Zero out padding tokens before pooling
        let padMaskExp = expandedDimensions(paddingPositions, axis: -1)
        let h = MLX.where(padMaskExp, MLXArray(Float(0)), hiddenStates)

        let length = outputLength ?? defaultOutputLength
        let pooled: MLXArray
        let mask: MLXArray
        if h.shape[1] == length {
            pooled = h
            mask = logicalNot(paddingPositions)
        } else {
            (pooled, mask) = avgPoolByPositions(h, patchPositions: patchPositions, length: length)
        }

        return (pooled * rootHiddenSize, mask)
    }
}

// MARK: - Vision Transformer Model (Encoder)

private class G4VisionTransformerModel: Module {
    @ModuleInfo var layers: [G4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            G4VisionTransformerBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        positions: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

// MARK: - Vision Model

private class G4VisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: G4VisionPatchEmbedder
    @ModuleInfo var encoder: G4VisionTransformerModel
    let pooler: G4VisionPooler
    let config: Gemma4VisionConfiguration

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        _patchEmbedder.wrappedValue = G4VisionPatchEmbedder(config: config)
        _encoder.wrappedValue = G4VisionTransformerModel(config: config)
        pooler = G4VisionPooler(config: config)
        super.init()
    }

    /// Compute patch positions for a single image of size H×W.
    /// Returns (patchPositions [maxPatches, 2] int32, paddingMask [maxPatches] bool).
    private func patchPositionsSingle(h: Int, w: Int) -> ([Int32], [Bool]) {
        let pH = h / config.patchSize
        let pW = w / config.patchSize
        let numRealPatches = pH * pW
        let maxPatches = config.maxPatches

        var posArray = [Int32](repeating: 0, count: maxPatches * 2)
        for row in 0 ..< pH {
            for col in 0 ..< pW {
                let idx = row * pW + col
                posArray[idx * 2] = Int32(col)  // x
                posArray[idx * 2 + 1] = Int32(row)  // y
            }
        }
        let numPad = maxPatches - numRealPatches
        if numPad > 0 {
            for i in numRealPatches ..< maxPatches {
                posArray[i * 2] = -1
                posArray[i * 2 + 1] = -1
            }
        }

        var padMask = [Bool](repeating: false, count: maxPatches)
        if numPad > 0 {
            for i in numRealPatches ..< maxPatches {
                padMask[i] = true
            }
        }

        return (posArray, padMask)
    }

    /// pixelValues: [B, C, H, W] (channel-first). Returns [1, numSoftTokens, hiddenSize].
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let numRealPatches = min((H / config.patchSize) * (W / config.patchSize), config.maxPatches)
        let maxPatches = config.maxPatches

        let (posArray, padMaskArray) = patchPositionsSingle(h: H, w: W)

        // Tile for batch
        let patchPosSingle = MLXArray(posArray).reshaped(maxPatches, 2)  // [maxPatches, 2]
        let patchPositions = repeated(patchPosSingle.expandedDimensions(axis: 0), count: B, axis: 0)  // [B, maxP, 2]
        var paddingPositions = MLXArray(padMaskArray).expandedDimensions(axis: 0)  // [1, maxP]
        paddingPositions = repeated(paddingPositions, count: B, axis: 0)  // [B, maxP]

        // Patch embedder receives only real patches
        let realPatchPositions = patchPositions[0..., 0 ..< numRealPatches, 0...]  // [B, numReal, 2]
        let realPaddingPositions = paddingPositions[0..., 0 ..< numRealPatches]  // [B, numReal]
        var inputsEmbeds = patchEmbedder(
            pixelValues, patchPositions: realPatchPositions, paddingPositions: realPaddingPositions)
        // [B, numReal, hiddenSize]

        // Pad to maxPatches with zeros
        let numPad = maxPatches - numRealPatches
        if numPad > 0 {
            let padEmbeds = MLXArray.zeros(
                [B, numPad, inputsEmbeds.shape[2]], dtype: inputsEmbeds.dtype)
            inputsEmbeds = concatenated([inputsEmbeds, padEmbeds], axis: 1)
        }

        // Build bidirectional attention mask [B, 1, maxP, maxP]
        let validMask = logicalNot(paddingPositions)  // [B, maxP] bool
        let vmQF = expandedDimensions(validMask, axis: 1).asType(.float32)  // [B, 1, maxP]
        let vmKF = expandedDimensions(validMask, axis: 2).asType(.float32)  // [B, maxP, 1]
        let combined = (vmQF * vmKF).asType(.bool)  // [B, maxP, maxP]
        let negInf = MLXArray(Float(-Float.infinity)).asType(inputsEmbeds.dtype)
        let zero = MLXArray(Float(0.0)).asType(inputsEmbeds.dtype)
        let attnMaskFull = MLX.where(combined, zero, negInf)  // [B, maxP, maxP]
        let attnMask = expandedDimensions(attnMaskFull, axis: 1)  // [B, 1, maxP, maxP]

        // Run transformer
        let hiddenStates = encoder(inputsEmbeds, positions: patchPositions, mask: .array(attnMask))

        // Pool
        let (pooled, poolMask) = pooler(
            hiddenStates, patchPositions: patchPositions, paddingPositions: paddingPositions)
        // poolMask: [B, defaultOutputLength], True = valid

        // Determine valid count and concatenate across batch
        var allReal: [MLXArray] = []
        for i in 0 ..< B {
            let maskRow = poolMask[i]  // [defaultOutputLength]
            let nValid = Int(maskRow.asType(DType.int32).sum().asArray(Int32.self)[0])
            allReal.append(pooled[i, 0 ..< nValid, 0...])
        }
        let result = concatenated(allReal, axis: 0)  // [totalSoftTokens, hiddenSize]
        return result.expandedDimensions(axis: 0)  // [1, totalSoftTokens, hiddenSize]
    }
}

// MARK: - Text Components (re-implemented for MLXVLM module isolation)

private class G4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class G4Attention: Module {
    let isSliding: Bool
    let isKvSharedLayer: Bool
    let useKEqV: Bool
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: G4VisionRMSNormNoScale  // RMSNormNoScale for v
    @ModuleInfo var rope: RoPE

    init(_ config: G4TextConfig, layerIdx: Int) {
        let layerTypes = config.effectiveLayerTypes
        isSliding = layerTypes[layerIdx] == "sliding_attention"

        let firstKvSharedLayerIdx = config.numHiddenLayers - (config.numKvSharedLayers ?? 0)
        isKvSharedLayer = (config.numKvSharedLayers ?? 0) > 0 && layerIdx >= firstKvSharedLayerIdx

        let useGlobalHeadDim = !isSliding && (config.globalHeadDim ?? 0) > 0
        headDim = useGlobalHeadDim ? (config.globalHeadDim ?? config.headDim) : config.headDim
        numHeads = config.numAttentionHeads

        useKEqV = (config.attentionKEqV ?? false) && !isSliding
        if useKEqV, let numGlobal = config.numGlobalKeyValueHeads {
            numKVHeads = numGlobal
        } else {
            numKVHeads = config.numKeyValueHeads
        }

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = G4VisionRMSNormNoScale(eps: config.rmsNormEps)

        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters?[layerKey]
        let ropeTheta: Float
        if let theta = ropeParams?["rope_theta"]?.asFloat() {
            ropeTheta = theta
        } else {
            ropeTheta = isSliding ? 10000.0 : 1_000_000.0
        }
        // partial_rotary_factor defaults to 1.0 (full rotation) if not specified
        let partialFactor = config.partialRotaryFactor ?? 1.0
        let rotaryDim = Int(Float(headDim) * partialFactor)
        _rope.wrappedValue = RoPE(dimensions: rotaryDim, traditional: false, base: ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, -1, headDim)
        queries = qNorm(queries)

        var keys: MLXArray
        var values: MLXArray

        // Capture offset BEFORE any cache.update() call (mirrors Python: offset = cache.offset + 0).
        // This ensures both keys and queries use the same pre-update offset for RoPE.
        let ropeOffset = cache?.offset ?? 0

        if isKvSharedLayer, let cache {
            let state = cache.state
            if state.count >= 2 {
                keys = state[0]
                values = state[1]
            } else {
                keys = kProj(x).reshaped(B, L, -1, headDim)
                if useKEqV { values = keys } else { values = vProj(x).reshaped(B, L, -1, headDim) }
                keys = kNorm(keys)
                values = vNorm(values)
                keys = keys.transposed(0, 2, 1, 3)
                keys = rope(keys, offset: ropeOffset)
                values = values.transposed(0, 2, 1, 3)
                (keys, values) = cache.update(keys: keys, values: values)
            }
        } else {
            keys = kProj(x).reshaped(B, L, -1, headDim)
            if useKEqV { values = keys } else { values = vProj(x).reshaped(B, L, -1, headDim) }
            keys = kNorm(keys)
            values = vNorm(values)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: ropeOffset)
            values = values.transposed(0, 2, 1, 3)
            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: ropeOffset)

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(keys.ndim - 2)
            if maskArray.shape.last! != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen].asType(queries.dtype))
            } else {
                adjustedMask = .array(maskArray.asType(queries.dtype))
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1.0, mask: adjustedMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

private class G4DecoderLayer: Module {
    let hasPerLayerInput: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: G4Attention
    @ModuleInfo var mlp: G4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    init(_ config: G4TextConfig, layerIdx: Int) {
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 0
        hasPerLayerInput = hiddenSizePerLayerInput > 0

        let firstKvSharedLayerIdx = config.numHiddenLayers - (config.numKvSharedLayers ?? 0)
        let isKvSharedLayer =
            (config.numKvSharedLayers ?? 0) > 0 && layerIdx >= firstKvSharedLayerIdx
        let useDoubleWide = (config.useDoubleWideMlp ?? false) && isKvSharedLayer
        let effectiveIntermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        _selfAttn.wrappedValue = G4Attention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = G4MLP(
            hiddenSize: config.hiddenSize, intermediateSize: effectiveIntermediateSize)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layerScalar.wrappedValue = MLXArray.ones([1])

        if hasPerLayerInput {
            _perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            _perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            _postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            _perLayerInputGate.wrappedValue = nil
            _perLayerProjection.wrappedValue = nil
            _postPerLayerInputNorm.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postAttentionLayernorm(h)
        h = residual + h

        residual = h
        h = preFeedforwardLayernorm(h)
        h = mlp(h)
        h = postFeedforwardLayernorm(h)
        h = residual + h

        if hasPerLayerInput,
            let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let pli = perLayerInput
        {
            residual = h
            var g = gate(h)
            g = geluApproximate(g)
            g = g * pli
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        return h * layerScalar
    }
}

private class G4TextModel: Module {
    let config: G4TextConfig
    let firstKvSharedLayerIdx: Int
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let layerIdxToCacheIdx: [Int]
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    let perLayerInputScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [G4DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: G4RMSNormZeroShift?

    init(_ config: G4TextConfig) {
        self.config = config
        embedScale = pow(Float(config.hiddenSize), 0.5)

        let numShared = config.numKvSharedLayers ?? 0
        firstKvSharedLayerIdx = config.numHiddenLayers - numShared
        let layerTypes = config.effectiveLayerTypes
        let concreteLayerTypes = Array(layerTypes.prefix(firstKvSharedLayerIdx))

        var idxToCacheIdx = Array(0 ..< firstKvSharedLayerIdx)
        if numShared > 0 {
            let sharedFullIdx =
                concreteLayerTypes.lastIndex(of: "full_attention") ?? (firstKvSharedLayerIdx - 1)
            let sharedSlidingIdx = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for i in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                idxToCacheIdx.append(
                    layerTypes[i] == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = idxToCacheIdx

        firstFullCacheIdx = concreteLayerTypes.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayerTypes.firstIndex(of: "sliding_attention") ?? 0

        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256
        let vocabSizePerLayerInput = config.vocabSizePerLayerInput ?? config.vocabularySize
        embedTokensPerLayerScale = pow(Float(hiddenSizePerLayerInput), 0.5)
        perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        perLayerInputScale = pow(2.0, -0.5)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            G4DecoderLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hasPerLayerInput {
            _embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * hiddenSizePerLayerInput
            )
            _perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * hiddenSizePerLayerInput,
                bias: false
            )
            _perLayerProjectionNorm.wrappedValue = G4RMSNormZeroShift(
                dimensions: hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            _embedTokensPerLayer.wrappedValue = nil
            _perLayerModelProjection.wrappedValue = nil
            _perLayerProjectionNorm.wrappedValue = nil
        }
        super.init()
    }

    /// Return scaled token embeddings.
    func scaledEmbedding(_ inputIds: MLXArray) -> MLXArray {
        var h = embedTokens(inputIds)
        h = h * MLXArray(embedScale, dtype: h.dtype)
        return h
    }

    /// Compute raw per-layer token inputs from (masked) token IDs.
    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else { fatalError("embedTokensPerLayer is nil") }
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256
        var result = embedTokensPerLayer(inputIds)
        result = result * MLXArray(embedTokensPerLayerScale, dtype: result.dtype)
        result = result.reshaped(
            Array(inputIds.shape) + [config.numHiddenLayers, hiddenSizePerLayerInput])
        return result
    }

    private func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?)
        -> MLXArray
    {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            fatalError("Per-layer projection modules are nil")
        }
        let hiddenSizePerLayerInput = config.hiddenSizePerLayerInput ?? 256

        var proj = perLayerModelProjection(inputsEmbeds)
        proj = proj * MLXArray(perLayerProjectionScale, dtype: inputsEmbeds.dtype)
        proj = proj.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.numHiddenLayers, hiddenSizePerLayerInput,
            ])
        proj = perLayerProjectionNorm(proj)

        guard let perLayerInputs else { return proj }
        return (proj + perLayerInputs) * MLXArray(perLayerInputScale, dtype: inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray?,
        inputsEmbeds: MLXArray? = nil,
        precomputedPerLayerInputs: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]?
    ) -> MLXArray {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else if let inputs {
            h = scaledEmbedding(inputs)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        var finalPerLayerInputs: MLXArray? = nil
        if config.hasPerLayerInput {
            if let pre = precomputedPerLayerInputs {
                finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: pre)
            } else if let inputs {
                let tokenPLI = getPerLayerInputs(inputs)
                finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: tokenPLI)
            }
        }

        let cacheArray: [KVCache?] = cache ?? Array(repeating: nil, count: firstKvSharedLayerIdx)
        let layerTypes = config.effectiveLayerTypes

        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            globalMask = mask
            slidingMask = mask
        } else {
            let fullCache =
                firstFullCacheIdx < cacheArray.count ? cacheArray[firstFullCacheIdx] : nil
            let slidingCache =
                firstSlidingCacheIdx < cacheArray.count ? cacheArray[firstSlidingCacheIdx] : nil
            globalMask = createAttentionMask(h: h, cache: fullCache)
            slidingMask = createAttentionMask(
                h: h, cache: slidingCache, windowSize: config.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            let cacheIdx = layerIdxToCacheIdx[i]
            let layerCache = cacheIdx < cacheArray.count ? cacheArray[cacheIdx] : nil
            let isGlobal = layerTypes[i] == "full_attention"
            let localMask = isGlobal ? globalMask : slidingMask

            var perLayerInput: MLXArray? = nil
            if let finalPerLayerInputs {
                perLayerInput = finalPerLayerInputs[0..., 0..., i, 0...]
            }

            h = layer(h, mask: localMask, cache: layerCache, perLayerInput: perLayerInput)
        }

        return norm(h)
    }
}

// MARK: - Language Model Wrapper

private class G4LanguageModel: Module {
    @ModuleInfo var model: G4TextModel
    let textConfig: G4TextConfig
    var kvHeads: [Int]

    init(_ config: G4TextConfig) {
        textConfig = config
        _model.wrappedValue = G4TextModel(config)
        kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let layerTypes = textConfig.effectiveLayerTypes
        var caches: [any KVCache] = []
        for i in 0 ..< textConfig.numCaches {
            if layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: textConfig.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    func callForLogits(
        _ inputs: MLXArray?,
        inputsEmbeds: MLXArray? = nil,
        precomputedPerLayerInputs: MLXArray? = nil,
        cache: [any KVCache]?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        let cacheArray = cache?.map { $0 as KVCache? }
        let out = model(
            inputs, inputsEmbeds: inputsEmbeds,
            precomputedPerLayerInputs: precomputedPerLayerInputs, mask: mask, cache: cacheArray)
        var logits = model.embedTokens.asLinear(out)
        if let cap = textConfig.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }
}

// MARK: - Multimodal Embedder

private class G4MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var preNorm: G4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float = 1e-6) {
        _embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        _preNorm.wrappedValue = G4RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(preNorm(x))
    }
}

// MARK: - Gemma4 VLM

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: G4VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: G4LanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: G4MultimodalEmbedder

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.textConfig.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        _visionTower.wrappedValue = G4VisionModel(config: config.visionConfig)
        _languageModel.wrappedValue = G4LanguageModel(config.textConfig)
        _embedVision.wrappedValue = G4MultimodalEmbedder(
            embeddingDim: config.visionConfig.hiddenSize,
            textHiddenSize: config.textConfig.hiddenSize,
            eps: config.visionConfig.rmsNormEps
        )
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel.callForLogits(inputs, cache: cache)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            // Text-only path
            let logits = languageModel.callForLogits(input.text.tokens, cache: cache)
            return .logits(LMOutput(logits: logits))
        }

        let inputIds = input.text.tokens

        // 1. Compute scaled text embeddings
        var inputsEmbeds = languageModel.model.scaledEmbedding(inputIds)
        let embedDim = inputsEmbeds.dim(2)

        // 2. Compute per-layer inputs with image tokens masked to 0
        var precomputedPerLayerInputs: MLXArray? = nil
        if config.textConfig.hasPerLayerInput {
            let imageTokenId = Int32(config.imageTokenId)
            let imageMask = inputIds .== MLXArray(imageTokenId)
            let maskedIds = MLX.where(imageMask, MLXArray(Int32(0)), inputIds)
            precomputedPerLayerInputs = languageModel.model.getPerLayerInputs(maskedIds)
        }

        // 3. Process vision: pixelValues [B, C, H, W] (channel-first from processor)
        let imageFeatures = visionTower(imagePixels)  // [1, numSoftTokens, visHidden]
        let projectedFeatures = embedVision(imageFeatures)  // [1, numSoftTokens, textHidden]

        // 4. Scatter image features into text embeddings at image token positions
        let imageMask = inputIds .== MLXArray(Int32(config.imageTokenId))
        let imageMaskExpanded = repeated(
            expandedDimensions(imageMask, axis: -1), count: embedDim, axis: -1)
        let scaledFeatures = projectedFeatures.asType(inputsEmbeds.dtype)
        inputsEmbeds = maskedScatterG4(
            inputsEmbeds, mask: imageMaskExpanded, source: scaledFeatures)

        // 5. Run language model with causal mask
        let logits = languageModel.callForLogits(
            nil,
            inputsEmbeds: inputsEmbeds,
            precomputedPerLayerInputs: precomputedPerLayerInputs,
            cache: cache,
            mask: .causal
        )
        return .logits(LMOutput(logits: logits))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var k = key
            // Strip outer "model." prefix
            if k.hasPrefix("model.") {
                k = String(k.dropFirst("model.".count))
            }
            // Skip rotary embeddings and clipping parameters
            if k.contains("rotary_emb") { continue }
            if k.contains("input_max") || k.contains("input_min")
                || k.contains("output_max") || k.contains("output_min")
            {
                continue
            }
            // Skip audio tower and audio embedder
            if k.hasPrefix("audio_tower") || k.hasPrefix("embed_audio") { continue }
            // Map language_model.X → language_model.model.X
            if k.hasPrefix("language_model.") && !k.hasPrefix("language_model.model.") {
                let rest = String(k.dropFirst("language_model.".count))
                k = "language_model.model." + rest
            }
            sanitized[k] = value
        }

        // Apply quantization: convert Linear/Embedding → Quantized variants.
        // VLMModelFactory only passes perLayerQuantization to loadWeights, so when
        // a model uses a top-level "quantization" key (as Gemma4 E4B does), the
        // framework-level quantization path is never triggered. We handle it here,
        // same pattern as Gemma3.LanguageModel.sanitize.
        if sanitized["language_model.model.layers.0.self_attn.q_proj.scales"] != nil {
            let q = config.quantization?.asTuple ?? (64, 4, .affine)
            quantize(model: self) { path, _ in
                if sanitized["\(path).scales"] != nil
                    && sanitized["\(path).weight"]?.dtype == .uint32
                {
                    return q
                }
                return nil
            }
        }

        return sanitized
    }

    public var loraLayers: [Module] { languageModel.model.layers }
}
