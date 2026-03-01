#include "models.h"
#include "llama-impl.h"

// EAGLE3_DS Encoder: processes target model features through feature fusion layer
// Same concept as eagle3 encoder but supports variable number of extraction layers (2 or 3)
llm_build_eagle3_ds_encode::llm_build_eagle3_ds_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int n_extract = hparams.eagle3_n_extract;
    // EAGLE v1/v2 & MTP: FC input = concat(embedding, hidden) = 2 * target_hidden_size
    // Eagle-3:           FC input = concat(layer_features) = n_extract * target_hidden_size
    const int64_t n_embd_target_features = (hparams.eagle_is_v1 || hparams.eagle_is_mtp)
        ? 2 * (int64_t)hparams.eagle3_target_hidden_size
        : n_extract * (int64_t)hparams.eagle3_target_hidden_size;

    ggml_tensor * cur = nullptr;

    // Input: Target model features (N layers concatenated)
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_target_features);
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings [n_embd, n_tokens]
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

// EAGLE3_DS Decoder: processes draft tokens using g_embeddings from encoder
// Uses DeepSeek V2 / Mistral style MLA attention + MoE FFN
llm_build_eagle3_ds_decode::llm_build_eagle3_ds_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // YaRN scaling (same as deepseek2)
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    GGML_ASSERT(n_layer == 1);  // EAGLE-3 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Choose token_embd: prefer EAGLE3's own if available, else use target's
    ggml_tensor * token_embd_eagle3 = (model.tok_embd != nullptr) ? model.tok_embd : model.target_tok_embd;

    // When loaded standalone (e.g. during params_fit/sched_reserve/warmup), target tensors are not yet available.
    // Build a minimal placeholder graph with proper output dimensions so graph reservation and decode can proceed.
    if (token_embd_eagle3 == nullptr || model.output == nullptr) {
        LLAMA_LOG_WARN("%s: EAGLE3_DS PLACEHOLDER graph (tok_embd=%p, output=%p, target_tok_embd=%p)\n",
                __func__, (void*)model.tok_embd, (void*)model.output, (void*)model.target_tok_embd);
        const int64_t n_vocab = model.vocab.n_tokens();

        ggml_tensor * dummy_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(dummy_embd);
        ggml_set_output(dummy_embd);
        res->t_embd = dummy_embd;

        // Logits must be [n_vocab, n_outputs] to match what llama_decode expects to read
        ggml_tensor * dummy_logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens);
        ggml_set_input(dummy_logits);
        ggml_set_output(dummy_logits);
        res->t_logits = dummy_logits;

        ggml_build_forward_expand(gf, dummy_embd);
        ggml_build_forward_expand(gf, dummy_logits);
        return;
    }

    LLAMA_LOG_INFO("%s: EAGLE3_DS REAL decoder graph (n_tokens=%d, tok_embd=%p, output=%p)\n",
            __func__, (int)n_tokens, (void*)token_embd_eagle3, (void*)model.output);

    ggml_tensor * inp_embd = build_inp_embd(token_embd_eagle3);
    cb(inp_embd, "inp_embd", -1);

    // g_embeddings input: encoder output (Eagle-3) or result_norm (EAGLE v1/v2)
    ggml_tensor * inp_g = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp_g);
    cb(inp_g, "inp_g_embeddings", -1);

    const bool eagle_v1 = hparams.eagle_is_v1;
    const bool eagle_mtp = hparams.eagle_is_mtp;

    if (eagle_mtp) {
        // MTP/NextN: enorm(embd) + hnorm(hidden) → concat → eh_proj → decoder input
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[0].nextn.enorm, NULL,
                LLM_NORM_RMS, -1);
        cb(embd_norm, "mtp_embd_norm", -1);

        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[0].nextn.hnorm, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "mtp_g_norm", -1);

        ggml_tensor * concat_input = ggml_concat(ctx0, embd_norm, g_norm, 0);  // [2*n_embd, n_tokens]
        cb(concat_input, "mtp_concat", -1);

        cur = build_lora_mm(model.fc, concat_input);  // eh_proj: [2*n_embd] -> [n_embd]
        cb(cur, "mtp_fc_out", -1);

        inpL = cur;  // residual from FC output
    } else if (eagle_v1) {
        // EAGLE v1/v2: FC(concat(embedding, hidden_state)) -> decoder input
        // inp_embd = token embeddings [n_embd, n_tokens]
        // inp_g    = final hidden state from target model [n_embd, n_tokens]
        ggml_tensor * concat_input = ggml_concat(ctx0, inp_embd, inp_g, 0);  // [2*n_embd, n_tokens]
        cb(concat_input, "eagle_concat", -1);

        cur = build_lora_mm(model.fc, concat_input);  // FC: [2*n_embd] -> [n_embd]
        cb(cur, "eagle_fc_out", -1);

        inpL = cur;  // residual from FC output
    } else {
        // Eagle-3: add(norm(embedding), norm(g_embeddings))
        inpL = inp_g;
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = is_mla ? build_attn_inp_k() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        if (eagle_v1 || eagle_mtp) {
            // EAGLE v1/v2 & MTP: apply attn_norm to the FC output (standard transformer pre-norm)
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "fused_embd", il);
        } else {
            // Eagle-3: apply input_layernorm to the token embeddings
            ggml_tensor * embd_norm = build_norm(inp_embd,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(embd_norm, "embd_norm", il);

            // Apply hidden_norm to inp_g (if available)
            ggml_tensor * g_norm;
            if (model.layers[il].eagle3_hidden_norm) {
                g_norm = build_norm(inp_g,
                        model.layers[il].eagle3_hidden_norm, NULL,
                        LLM_NORM_RMS, -1);
            } else {
                g_norm = build_norm(inp_g,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, -1);
            }
            cb(g_norm, "g_norm", il);

            // Fuse token embeddings and g_embeddings via element-wise addition -> [n_embd, n_tokens]
            cur = ggml_add(ctx0, embd_norm, g_norm);
            cb(cur, "fused_embd", il);
        }

        ggml_tensor * inpSA = inpL;

        // MLA self-attention on fused input
        {
            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(q, "q", il);

            q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q", il);

            q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                // MLA with absorption optimization -> MQA
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            } else {
                // Non-MLA path (fallback)
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
                cb(k_nope, "k_nope_view", il);

                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur_view", il);
                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(Kcur, "Kcur", il);

                auto * inp_attn_kv_fallback = build_attn_inp_kv();
                cur = build_attn(inp_attn_kv_fallback,
                            model.layers[il].wo, NULL,
                            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }
        }

        if (inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Add residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN norm
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // MoE FFN
        ggml_tensor * moe_out = build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            model.layers[il].ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale, hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
        cb(moe_out, "ffn_moe_out", il);

        // Shared expert (if available)
        if (model.layers[il].ffn_gate_shexp) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }

        // Output with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    if (eagle_v1 && !eagle_mtp) {
        // EAGLE v1/v2: output post-norm state for autoregressive g_embeddings
        // Must match target model's result_norm (post-norm) for consistent FC input
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        ggml_set_output(cur);
        res->t_embd = cur;
    } else {
        // Eagle-3 / MTP: output prenorm state (for next token's g_embeddings)
        // MTP: hnorm expects pre-norm hidden state
        ggml_set_output(cur);
        res->t_embd = cur;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
    }

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
