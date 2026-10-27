# infantTorch!

We are building infantTorch to learn how to code in cuda and potentially beat PyTorch, hence the name infant torch.

For the transformer architecture, we are going off of ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).
There is also CNN code in this repo. We're going to make a great torch library and eventually implement everything!

this readme is under construction!

## Flow of a transformer for reference
### A. Encoder
1. Input
2. Input embedding
3. Positional encoding
4. Transformer (repeat N times)
    1. Multihead attention
    2. Addition + Norm
    3. Feed forward 
    4. Addition + Norm again
5. output to decoder

### B. Decoder
1. Output tokens so far
2. Output embedding
3. Positional encoding
4. Transformer decoder block (repeat N times)
    1. Masked multihead (prevents trying to see future tokens)
    2. Addition + Norm
    3. Multi-head cross attention!
        - queries come from the decoder, keys and values come from the encoder output
    4. Addition + Norm
    5. Feed-forward
    6. Addition + Norm
5. Linear projection (converts vectors into vocab scores)
6. Softmax (converts scores into next probability of token)
