#include "models.h"

// Qwen3-ASR chunking constants (from model config: n_window=50)
static const int QWEN3A_CHUNK_FRAMES = 100; // n_window * 2 = 50 * 2
static const int QWEN3A_TOKENS_PER_FULL_CHUNK = 13; // 100 frames -> 3 conv2d stride-2 -> 13 tokens

// Compute conv2d output size: floor((n + 2*p - k) / s) + 1
static int conv2d_out_size(int n, int k = 3, int s = 2, int p = 1) {
    return (n + 2 * p - k) / s + 1;
}

// Compute number of tokens for a chunk of given frame length after 3 conv2d layers
static int tokens_for_chunk(int n_frames) {
    int t = n_frames;
    for (int i = 0; i < 3; i++) {
        t = conv2d_out_size(t);
    }
    return t;
}

ggml_cgraph * clip_graph_qwen3a::build() {
    // Qwen3-ASR audio encoder with CHUNKED conv2d processing
    //
    // The encoder splits the mel spectrogram into chunks of 100 frames (n_window*2),
    // processes each chunk independently through 3 conv2d layers, then concatenates
    // valid tokens and runs a global transformer encoder.
    //
    // This chunking is essential for correctness because the conv2d boundary effects
    // (padding at chunk edges) produce different representations than processing
    // the full mel at once. PyTorch verification shows chunked conv + global attention
    // achieves cosine similarity 1.0 with the reference implementation.

    const int n_frames = img.nx;
    printf("DEBUG build: n_frames=%d n_mel=%d\n", img.nx, img.ny);
    const int n_mel    = img.ny;

    // Compute chunk parameters
    const int n_chunks = (n_frames + QWEN3A_CHUNK_FRAMES - 1) / QWEN3A_CHUNK_FRAMES;
    const int last_chunk_frames = (n_frames % QWEN3A_CHUNK_FRAMES == 0)
        ? QWEN3A_CHUNK_FRAMES
        : (n_frames % QWEN3A_CHUNK_FRAMES);
    const int last_chunk_tokens = tokens_for_chunk(last_chunk_frames);
    const int total_tokens = (n_chunks - 1) * QWEN3A_TOKENS_PER_FULL_CHUNK + last_chunk_tokens;

    // Input: all chunks padded to QWEN3A_CHUNK_FRAMES, stacked as batch
    // Shape: [QWEN3A_CHUNK_FRAMES, n_mel, 1, n_chunks]
    // The data feeding code will split the mel into chunks and pad the last one
    ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
        QWEN3A_CHUNK_FRAMES, n_mel, 1, n_chunks);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    // Conv2d block: process all chunks in batch
    // ggml conv2d: a=[KW,KH,IC,OC], b=[IW,IH,IC,N] -> result=[OW,OH,OC,N]
    // Our inp: ne=[100, 128, 1, n_chunks] which is [IW=100, IH=128, IC=1, N=n_chunks]
    {
        // Conv2d(1, 480, 3, stride=2, padding=1) + GELU
        printf("DEBUG n_embd=%d\n", n_embd);
        printf("DEBUG conv2d_1: kernel ne=[%ld,%ld,%ld,%ld] input ne=[%ld,%ld,%ld,%ld]\n",
            model.conv2d_1_w->ne[0], model.conv2d_1_w->ne[1], model.conv2d_1_w->ne[2], model.conv2d_1_w->ne[3],
            inp->ne[0], inp->ne[1], inp->ne[2], inp->ne[3]);
        inp = ggml_conv_2d(ctx0, model.conv2d_1_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_1_b);
        inp = ggml_gelu_erf(ctx0, inp);

        // Conv2d(480, 480, 3, stride=2, padding=1) + GELU
        inp = ggml_conv_2d(ctx0, model.conv2d_2_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_2_b);
        inp = ggml_gelu_erf(ctx0, inp);

        // Conv2d(480, 480, 3, stride=2, padding=1) + GELU
        inp = ggml_conv_2d(ctx0, model.conv2d_3_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_3_b);
        inp = ggml_gelu_erf(ctx0, inp);
        cb(inp, "after_conv_blocks", -1);

        // After 3 conv2d with stride=2 on [100, 128, 1, n_chunks]:
        // inp shape (ggml ne): [OW=13, OH=16, OC=480, N=n_chunks]
        // = [time_tokens_per_chunk, freq_bins, channels, n_chunks]

        // Permute to [time, freq, channels, n_chunks] -> [time, channels*freq, n_chunks]
        // Current: ne = [13, 16, 480, n_chunks]
        // Need: [time, channels, freq, n_chunks] then flatten channels*freq
        // Permute (2, 1, 0, 3): ne[0]=480, ne[1]=16, ne[2]=13, ne[3]=n_chunks
        //   -> this gives [channels, freq, time, n_chunks]
        // Then permute again or reshape differently

        // Actually, let's match PyTorch's permute(0, 3, 1, 2) on [B, C, F, T]:
        // PyTorch BCFT -> BTCF -> reshape to [B, T, C*F]
        // In ggml ne order, our tensor is [T=13, F=16, C=480, B=n_chunks]
        // We want [C*F, T, B] for the linear projection (mul_mat expects [out, in] x [in, seq])
        // After permute we want ne = [C*F, T, B] = [7680, 13, n_chunks]
        //
        // Step 1: Permute [T, F, C, B] -> [F, C, T, B] via permute(1, 2, 0, 3)
        // Then flatten F*C to get [F*C, T, B]
        inp = ggml_permute(ctx0, inp, 2, 1, 0, 3);  // [T,F,C,B] -> [F,C,T,B] => flatten to [FC,T,B]
        inp = ggml_cont(ctx0, inp);

        // Flatten freq * channels dimensions
        inp = ggml_reshape_3d(ctx0, inp,
            16 * 480,                   // ne[0] = F*C = 7680
            QWEN3A_TOKENS_PER_FULL_CHUNK, // ne[1] = T = 13
            n_chunks);                   // ne[2] = B

        // Linear projection to d_model: conv_out weight is [d_model, 7680]
        // mul_mat: [d_model, 7680] x [7680, T*B] -> [d_model, T*B]
        // We need to flatten T and B for mul_mat, then reshape back
        inp = ggml_reshape_2d(ctx0, inp, 16 * 480, QWEN3A_TOKENS_PER_FULL_CHUNK * n_chunks);
        cb(inp, "before_mul_mat", -1);
        inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
        cb(inp, "after_mul_mat", -1);
        if (model.conv_out_b) {
            inp = ggml_add(ctx0, inp, model.conv_out_b);
        }

        // Reshape back to [d_model, T, B]
        inp = ggml_reshape_3d(ctx0, inp, n_embd, QWEN3A_TOKENS_PER_FULL_CHUNK, n_chunks);
        cb(inp, "after_conv_out", -1);
    ggml_set_output(inp);
    }

    // Add positional embeddings (same for each chunk, broadcast over batch dim)
    // Position embeddings: [d_model, max_pos]
    // We need [d_model, T] where T = tokens per chunk = 13
    ggml_tensor * pos_embd = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], QWEN3A_TOKENS_PER_FULL_CHUNK,
        model.position_embeddings->nb[1], 0
    );
    // This adds pos_embd [d_model, 13] to inp [d_model, 13, n_chunks] with broadcasting
    inp = ggml_add(ctx0, inp, pos_embd);
    cb(inp, "after_pos_embd", -1);

    // Extract valid tokens from all chunks and concatenate
    // For full chunks (0 to n_chunks-2): all 13 tokens are valid
    // For last chunk: only last_chunk_tokens are valid
    //
    // If last chunk is full (last_chunk_tokens == 13), we can simply reshape
    // Otherwise, we need to extract valid tokens from each chunk

    ggml_tensor * flat;
    if (last_chunk_tokens == QWEN3A_TOKENS_PER_FULL_CHUNK) {
        // All chunks are full - simple reshape to [d_model, total_tokens]
        flat = ggml_reshape_2d(ctx0, inp, n_embd, total_tokens);
    } else {
        // Need to extract valid tokens
        // Full chunks: [d_model, 13, n_chunks-1] -> [d_model, 13*(n_chunks-1)]
        if (n_chunks > 1) {
            ggml_tensor * full_chunks = ggml_view_3d(ctx0, inp,
                n_embd, QWEN3A_TOKENS_PER_FULL_CHUNK, n_chunks - 1,
                inp->nb[1], inp->nb[2], 0);
            full_chunks = ggml_reshape_2d(ctx0, full_chunks, n_embd,
                QWEN3A_TOKENS_PER_FULL_CHUNK * (n_chunks - 1));

            // Last chunk: [d_model, last_chunk_tokens]
            ggml_tensor * last_chunk = ggml_view_2d(ctx0, inp,
                n_embd, last_chunk_tokens,
                inp->nb[1],
                inp->nb[2] * (n_chunks - 1));

            // Concatenate: [d_model, total_tokens]
            flat = ggml_concat(ctx0, full_chunks, last_chunk, 1);
        } else {
            // Only one chunk (short audio)
            flat = ggml_view_2d(ctx0, inp,
                n_embd, last_chunk_tokens,
                inp->nb[1], 0);
        }
    }
    flat = ggml_cont(ctx0, flat);
    cb(flat, "flat_tokens", -1);
    ggml_set_output(flat);

    // Create block-diagonal windowed attention mask
    // Window size: tokens_per_chunk * (n_window_infer / chunk_size) = 13 * 8 = 104
    const int window_size = QWEN3A_TOKENS_PER_FULL_CHUNK * 8;  // 104

    // Create mask tensor [total_tokens, total_tokens]
    // Values: 0.0 for tokens in same window, -inf for tokens in different windows
    ggml_tensor * window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, total_tokens, total_tokens);
    ggml_set_name(window_mask, "window_mask");
    ggml_set_input(window_mask);

    // Whisper-like transformer encoder with WINDOWED attention
    ggml_tensor * cur = build_vit(
                            flat, total_tokens,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr, // pos embeddings already added above
                            nullptr, // no add_pos callback
                            window_mask); // block-diagonal windowed attention
    cb(cur, "after_transformer", -1);

    // Projector: proj1 -> GELU -> proj2
    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
