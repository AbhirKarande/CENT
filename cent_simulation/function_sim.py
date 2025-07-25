import torch
from utils import get_args, compare, n_heads, gqa_factor, embedding_size, ffn_size

# Mapping from argparse model choices to dictionary keys
MODEL_NAME_MAP = {
    "llama-2-7b": "Llama2-7B",
    "llama-2-13b": "Llama2-13B",
    "llama-2-70b": "Llama2-70B",
}

def create_dummy_dic_model(args):
    """
    Creates a dictionary with dummy tensors for model weights and inputs,
    allowing the simulation to run without loading a real model file.
    """
    model_name = MODEL_NAME_MAP.get(args.model)
    if not model_name:
        raise ValueError(f"Model '{args.model}' is not supported for dummy data generation.")

    dim = embedding_size[model_name]
    nh = n_heads[model_name]
    n_kv_h = nh // gqa_factor[model_name]
    head_dim = dim // nh
    ffn_dim = ffn_size[model_name]
    
    bsz = 1
    seqlen = 1
    start_pos = args.seqlen - 1 if args.seqlen > 0 else 0
    max_seq_len = args.max_seq_len

    dic_model = {
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(nh),
        "n_kv_heads": torch.tensor(n_kv_h),
        "start_pos": start_pos,
    }

    # --- Inputs and Weights (Randomly Initialized) ---
    dic_model["x"] = torch.randn(bsz, seqlen, dim)
    dic_model["SANorm"] = torch.randn(dim)
    dic_model["FFNNorm"] = torch.randn(dim)
    dic_model["freqs_cis"] = torch.randn(max_seq_len, head_dim)

    dic_model["wq"] = torch.randn(nh * head_dim, dim)
    dic_model["wk"] = torch.randn(n_kv_h * head_dim, dim)
    dic_model["wv"] = torch.randn(n_kv_h * head_dim, dim)
    dic_model["wo"] = torch.randn(dim, nh * head_dim)
    dic_model["w1"] = torch.randn(ffn_dim, dim)
    dic_model["w2"] = torch.randn(dim, ffn_dim)
    if "Llama" in model_name:
        dic_model["w3"] = torch.randn(ffn_dim, dim)

    # --- Caches and Dummy Tensors for Initialization ---
    dic_model["cache_k"] = torch.zeros(bsz, max_seq_len, n_kv_h, head_dim)
    dic_model["cache_v"] = torch.zeros(bsz, max_seq_len, n_kv_h, head_dim)
    
    # Dummy tensors for fields that are normally ground truth outputs
    dic_model["xq"] = torch.empty(bsz, seqlen, nh * head_dim)
    dic_model["xk"] = torch.empty(bsz, seqlen, n_kv_h * head_dim)
    dic_model["xv"] = torch.empty(bsz, seqlen, n_kv_h * head_dim)
    dic_model["scores"] = torch.empty(bsz, nh, seqlen, start_pos + seqlen)
    dic_model["output"] = torch.empty(bsz, seqlen, nh * head_dim)
    dic_model["sa"] = torch.empty(bsz, seqlen, dim)
    dic_model["h"] = torch.empty(bsz, seqlen, dim)
    dic_model["ffn"] = torch.empty(bsz, seqlen, dim)
    dic_model["out"] = torch.empty(bsz, seqlen, dim)

    return dic_model

if __name__ == "__main__":
    
    args = get_args()

    # Infer model parameters from the model name if not provided
    if args.model and (args.n_heads is None or args.ffn_dim is None):
        model_key = MODEL_NAME_MAP.get(args.model)
        if model_key:
            if args.n_heads is None:
                args.n_heads = n_heads[model_key]
            if args.ffn_dim is None:
                args.ffn_dim = ffn_size[model_key]

    if args.Llama or args.Llama_GQA:
        from Llama import TransformerBlockLlama as TransformerBlock
    elif args.BLOOM or args.OPT_66B or args.GPT3_175B or args.GPT3_175B_TP_8:
        from GPT import TransformerBlockGPT as TransformerBlock
    else:
        raise ValueError("A model type must be specified (--Llama, --Llama-GQA, etc.)")

    if args.pim_memory_mapping:
        if args.no_weights:
            print("Running with randomly generated weights and inputs.")
            dic_model = create_dummy_dic_model(args)
            TB = TransformerBlock(dic_model, args)
            TB.memory_mapping()
        else:
            if not args.filename:
                raise ValueError("--filename must be provided when --no-weights is not used.")
            print(f"Loading model from {args.filename}")
            dic_model = torch.load(args.filename)
            TB = TransformerBlock(dic_model, args)
            TB.memory_mapping()
            TB.memory_mapping_verification()

        print("\n============ Functional Simulation ============")
        sa_aim = TB.self_attention_disaggregated()
        out_aim = TB.FFN_disaggregated(sa_aim)

        if args.no_weights:
            print("Disaggregated functions executed successfully.")
            print(f"Final output shape: {out_aim.shape}")
        else:
            compare(out_aim[0][0], TB.out[0][0], "AiM out")

        TB.finish()
        TB.file.close()

    elif args.only_trace:
        if args.no_weights:
             dic_model = create_dummy_dic_model(args)
        else:
            dic_model = torch.load(args.filename)
        TB = TransformerBlock(dic_model, args)
        TB.memory_mapping()
        if args.embedding:
            TB.trace_only_embedding()
        elif args.only_FC:
            TB.trace_only_FC()
        else:
            TB.trace_only()
        TB.finish()
        TB.file.close()

    else:
        print("Please specify a simulation mode: --pim-memory-mapping or --only-trace")
