// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

public class VLMEvalTests: XCTestCase {

    // MARK: - Gemma4 Text-Only Forward Pass

    func testGemma4TextEval() throws {
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "vocab_size": 200,
                "num_key_value_heads": 1,
                "sliding_window": 8,
                "sliding_window_pattern": 2,
                "rms_norm_eps": 1e-6
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "patch_size": 4,
                "position_embedding_size": 64,
                "pooling_kernel_size": 2,
                "default_output_length": 4,
                "rms_norm_eps": 1e-6
            }
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 200])
    }

    // MARK: - Gemma4 Vision + Text Forward Pass

    func testGemma4VisionEval() throws {
        // patchSize=4, poolingKernelSize=2, defaultOutputLength=4
        // maxPatches = 4 * 2 * 2 = 16
        // Use a 2×2 pool grid → 4 output tokens, each 2×2 kernel → 16 patches total
        // Image: pH=4, pW=4 patches → H=16, W=16 pixels
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "vocab_size": 200,
                "num_key_value_heads": 1,
                "sliding_window": 8,
                "sliding_window_pattern": 2,
                "rms_norm_eps": 1e-6
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "patch_size": 4,
                "position_embedding_size": 64,
                "pooling_kernel_size": 2,
                "default_output_length": 4,
                "rms_norm_eps": 1e-6
            }
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4(config)

        // 4 image tokens in the prompt (matching 4 soft tokens from the image)
        // imageTokenId = 258880, text tokens are small ints
        let numSoftTokens = 4
        let imageTokenId = config.imageTokenId
        let imageTokens = MLXArray(Array(repeating: Int32(imageTokenId), count: numSoftTokens))
        let textTokens = MLXArray([Int32(1), Int32(2), Int32(3)])
        let promptTokens = concatenated([textTokens, imageTokens, MLXArray([Int32(4), Int32(5)])])
            .expandedDimensions(axis: 0)  // [1, 10]

        // Synthetic pixel values: [1, 3, 16, 16] channel-first, float32
        let pixels = MLXRandom.uniform(low: 0.0, high: 1.0, [1, 3, 16, 16])
        let frames = [THW(1, 16, 16)]
        let processedImage = LMInput.ProcessedImage(pixels: pixels, frames: frames)

        let lmInput = LMInput(
            text: .init(tokens: promptTokens, mask: ones(like: promptTokens).asType(.int8)),
            image: processedImage
        )

        let cache = model.newCache(parameters: nil)
        let result = try model.prepare(lmInput, cache: cache, windowSize: nil)

        if case .logits(let lmOutput) = result {
            let logits = lmOutput.logits
            // Sequence length = 3 text + 4 image + 2 text = 9 non-image + numSoftTokens
            // But image tokens are replaced: prompt is 10 total, shape [1, 10, vocab_size]
            XCTAssertEqual(logits.shape[0], 1)
            XCTAssertEqual(logits.shape[1], promptTokens.shape[1])
            XCTAssertEqual(logits.shape[2], 200)
        } else {
            XCTFail("Expected .logits result from prepare()")
        }
    }
}
