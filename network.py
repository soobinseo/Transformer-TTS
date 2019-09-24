from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparams as hp
import copy

class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(t.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)
        self.layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)

    def forward(self, x, pos):

        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(t.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        # Encoder pre-network
        x = self.encoder_prenet(x)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask, attns


class MelDecoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden
        """
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.norm = Linear(num_hidden, num_hidden)

        self.selfattn_layers = clones(Attention(num_hidden), 3)
        self.dotattn_layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')

        self.postconvnet = PostConvNet(num_hidden)

    def forward(self, memory, decoder_input, c_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)

        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(t.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask = None, None

        # Decoder pre-network
        decoder_input = self.decoder_prenet(decoder_input)

        # Centered position
        decoder_input = self.norm(decoder_input)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        decoder_input = pos * self.alpha + decoder_input

        # Positional dropout
        decoder_input = self.pos_dropout(decoder_input)

        # Attention decoder-decoder, encoder-decoder
        attn_dot_list = list()
        attn_dec_list = list()

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)

        # Mel linear projection
        mel_out = self.mel_linear(decoder_input)
        
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)

        # Stop tokens
        stop_tokens = self.stop_linear(decoder_input)

        return mel_out, out, attn_dot_list, stop_tokens, attn_dec_list


class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = MelDecoder(hp.hidden_size)

    def forward(self, characters, mel_input, pos_text, pos_mel):
        memory, c_mask, attns_enc = self.encoder.forward(characters, pos=pos_text)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder.forward(memory, mel_input, c_mask,
                                                                                             pos=pos_mel)

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)

        return mag_pred