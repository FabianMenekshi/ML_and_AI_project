import torch
from torchinfo import summary
from train_gpt_block_attnres import GPT, Hyperparameters

args = Hyperparameters()

model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    decoder_attnres_num_blocks = 4
)

summary(model, input_size=(1, args.train_seq_len), dtypes=[torch.long])