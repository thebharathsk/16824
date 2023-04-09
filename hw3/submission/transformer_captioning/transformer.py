# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
       
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Initialize the following layers and parameters to perform attention
        # This class assumes that the input dimension for query, key and value is embed_dim
        #MY IMPLEMENTATION
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)
            
    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape
       
        # TODO : Compute attention 
        #MY IMPLEMENTATION
    
        #project query, key and value  - 
        query = self.query_proj(query) #NxSxD
        key = self.key_proj(key) #NxTxD
        value = self.value_proj(value) #NxTxD

        #compute dot-product attention. Don't forget the scaling value!
        #Expected shape of dot_product is (N, S, T)
        dot_product = torch.matmul(query, key.mT) #NxSxT

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = attn_mask #SxT
            additive_mask[additive_mask == 0] = -torch.inf 
            additive_mask[additive_mask == 1] = 0
            
            dot_product += additive_mask.unsqueeze(0) #NxSxT

        # apply softmax, dropout, and use value
        #apply scaling
        dot_product = dot_product/math.sqrt(D) #NxSxT
        
        #apply softmax
        y = F.softmax(dot_product, dim=2) #NxSxT
        
        #apply dropout
        y = self.dropout(y) #NxSxT
        
        #multiply values
        output = torch.matmul(y, value) #NxSxD
        
        return output 

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads
        
        # TODO: Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # TODO : Compute multi-head attention
        #MY IMPLEMENTATION
        
        #project query, key and value
        query = self.query_proj(query) #NxSxD
        key = self.key_proj(key) #NxTxD
        value = self.value_proj(value) #NxTxD
        
        #after projection, split the embedding across num_heads
        #eg - expected shape for value is (N, H, T, D/H)
        query = query.view(N, S, H, D//H).transpose(1, 2) #NxHxSxD//H
        key = key.view(N, T, H, D//H).transpose(1, 2) #NxHxTxD//H
        value = value.view(N, T, H, D//H).transpose(1, 2) #NxHxTxD//H
        
        #compute dot-product attention separately for each head. Don't forget the scaling value!
        #Expected shape of dot_product is (N, H, S, T)
        dot_product = torch.matmul(query, key.transpose(2, 3))/math.sqrt(D/H) #NxHxSxT

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = copy.deepcopy(attn_mask) #SxT
            additive_mask[additive_mask == 0] = -torch.inf 
            additive_mask[additive_mask == 1] = 0
            dot_product += additive_mask.unsqueeze(0).unsqueeze(1) #NxHxSxT
                
        # apply softmax, dropout, and use value
        y = F.softmax(dot_product, dim=3) #NxHxSXT
        
        #apply dropout
        y = self.dropout(y) #NxHxSXT
        
        #multiple values
        y = torch.matmul(y, value) #NxHxSxD/H
        
        # concat embeddings from different heads, and project
        #concatenate
        y = y.transpose(1, 2) #NxSxHxD/H
        y = y.contiguous().view(N, S, D) #NxSxD
        
        #project
        output = self.head_proj(y) #NxSxD
                
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        #MY IMPLEMENTATION
        # TODO - use torch.nn.Embedding to create the encoding. Initialize dropout layer.
        self.encoding = nn.Embedding(max_len, embed_dim)#, _freeze=True) 
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x):
        N, S, D = x.shape
        # TODO - add the encoding to x
        #MY IMPLEMENTATION
        #indices
        idx = torch.arange(0, S, device=x.device).unsqueeze(0) #1xS
        idx = torch.cat([idx]*N, dim=0).long() #NxS
        
        #get and add encodings
        enc_weights = self.encoding(idx) #NxSxD
        
        output = x + enc_weights #NxSxD
        
        output = self.dropout(output) #NxSxD
   
        return output


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for self_attn.
        #MY IMPLEMENTATION
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, mask):
        ############# TODO - Self-attention on the sequence, using the mask. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        #MY IMPLEMENTATION
        #attention
        y = self.self_attn(seq, seq, seq, mask)
        
        #dropout
        y = self.dropout(y)
        
        #residual connection and normalization
        out = self.layernorm(seq + y)
                
        return out

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for cross_attn.
        #MY IMPLEMENTATION
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, cond):
        ############# TODO - Cross-attention on the sequence, using conditioning. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        #MY IMPLEMENTATION
        #replicate cond
        cond_cat = torch.cat([cond]*seq.shape[1], dim=1)
        
        #attention
        y = self.cross_attn(seq, cond_cat, cond_cat)
        #y = self.cross_attn(cond_cat, seq, seq)
        
        #dropout
        y = self.dropout(y)
        
        #residual connection and normalization
        out = self.norm(seq + y)
                
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        # TODO: Initialize the following. 
        # MLP has the following layers : linear, relu, dropout, linear ; hidden dim of linear is given by dim_feedforward
        #MY IMPLEMENTATION
        self.mlp = nn.Sequential(nn.Linear(input_dim, dim_feedforward),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, input_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq):
         ############# TODO - MLP on the sequence. Add dropout to mlp layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        #MY IMPLEMENTATION
        #mlp
        y = self.mlp(seq)
        
        #dropout
        y = self.dropout(y)
        
        #residual and layer norm
        out = self.norm(seq + y)
        
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):        
        out = self.self_atn_block(seq, mask)
        
        out = self.cross_atn_block(out, cond)
        
        out = self.feedforward_block(out)
        
        return out
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        # TODO - get caption and feature embeddings 
        # Don't forget position embeddings for captions!
        # expected caption embedding output shape : (N, T, D)
        #MY IMPLEMENTATION
        #get embeddings
        caption_embedding = self.caption_embedding(captions) #NxTxD
        
        #add positional embeddings
        caption_embedding = self.positional_encoding(caption_embedding) #NxTxD
        
        # Unsqueeze feature embedding along dimension 1
        # expected feature embedding output shape : (N, 1, D)
        #compute features
        feature_embedding = self.feature_embedding(features)
        
        #unsqueeze
        feature_embedding = feature_embedding.unsqueeze(1)
        
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        #TODO - get causal mask. This should be a matrix of shape (_len, _len). 
        # This mask is multiplicative
        # setting mask[i,j] = 0 means jth element of the sequence is not used 
        # to predict the ith element of the sequence.
        #MY IMPLEMENTATION
        #create mask
        mask = torch.zeros((_len, _len)).to(self.device)
        
        #create meshgrid
        x = torch.arange(0, _len)
        xx, yy = torch.meshgrid(x, x)
        
        #set elements to 1
        mask[xx >= yy] = 1

        return mask
                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions