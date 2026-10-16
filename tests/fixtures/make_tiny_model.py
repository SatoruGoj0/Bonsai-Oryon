#!/usr/bin/env python3
"""Create a tiny synthetic llama-architecture GGUF (F32, random weights) for functional CI tests.

Shapes are chosen so the PQ2_0 kernels are exercised on every path once the file is quantized with
`llama-quantize ... PQ2_0`: n_embd = 256 (rows of 256 -> Q8_K activations) and n_ff = 640 (ffn_down rows
of 640 = 5 x 128, not a multiple of 256 -> Q8_0 fallback). The vocabulary is a small SentencePiece-style vocab
(unknown characters map to <unk>). No real model data is involved.

usage: make_tiny_model.py <gguf-py dir> <out.gguf>
"""
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import gguf  # noqa: E402

out = sys.argv[2]
rng = np.random.default_rng(20261016)

n_embd, n_head, n_head_kv, n_ff, n_layer, n_ctx = 256, 4, 2, 640, 2, 2048
head_dim = n_embd // n_head

# vocabulary: <unk> <s> </s>, then word pieces and single characters. No byte-fallback tokens on
# purpose: a random model would emit arbitrary byte sequences (invalid UTF-8), which the server's
# output parser rightly rejects; unknown input characters map to <unk>.
# (SPM vocabularies must contain the newline byte token, so <0x0A> is the one byte token kept.)
# ChatML control tokens are included so the chat template renders to real special tokens.
pieces = ["<unk>", "<s>", "</s>", "<0x0A>", "<|im_start|>", "<|im_end|>"]
types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL, gguf.TokenType.BYTE,
         gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]
words = ("the of and to in is that it for on with as was be by this are at from or an have not but "
         "which one all were they their there can more has been we if will would so what about out up "
         "model bonsai oryon tree small fast token text hello world test server api chat answer").split()
for w in words:
    pieces.append("▁" + w)
    types.append(gguf.TokenType.NORMAL)
for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()-":
    pieces.append(c)
    types.append(gguf.TokenType.NORMAL)
    pieces.append("▁" + c)
    types.append(gguf.TokenType.NORMAL)
pieces.append("▁")
types.append(gguf.TokenType.NORMAL)
n_vocab = len(pieces)
scores = [0.0] * 6 + [-float(i) / 10 for i in range(n_vocab - 6)]

w = gguf.GGUFWriter(out, "llama")
w.add_name("bonsai-oryon-tiny-test")
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_rope_dimension_count(head_dim)
w.add_rope_freq_base(10000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_vocab_size(n_vocab)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_tokenizer_model("llama")
w.add_tokenizer_pre("default")
w.add_token_list(pieces)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_unk_token_id(0)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_eot_token_id(5)
w.add_add_bos_token(True)
w.add_add_eos_token(False)
# standard ChatML template (recognized by llama.cpp's chat-format detection)
w.add_chat_template(
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")


def mat(rows, cols):
    # ggml shape [cols, rows] -> numpy (rows, cols); scaled so activations stay O(1)
    return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)


w.add_tensor("token_embd.weight", mat(n_vocab, n_embd))
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
w.add_tensor("output.weight", mat(n_vocab, n_embd))
for i in range(n_layer):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor(p + "attn_q.weight", mat(n_embd, n_embd))
    w.add_tensor(p + "attn_k.weight", mat(n_head_kv * head_dim, n_embd))
    w.add_tensor(p + "attn_v.weight", mat(n_head_kv * head_dim, n_embd))
    w.add_tensor(p + "attn_output.weight", mat(n_embd, n_embd))
    w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor(p + "ffn_gate.weight", mat(n_ff, n_embd))
    w.add_tensor(p + "ffn_up.weight", mat(n_ff, n_embd))
    w.add_tensor(p + "ffn_down.weight", mat(n_embd, n_ff))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out}: vocab={n_vocab} embd={n_embd} ff={n_ff} layers={n_layer}")
