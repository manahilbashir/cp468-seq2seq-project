"""
LSTM-based Encoder-Decoder with Bahdanau Attention.

Built from scratch using PyTorch standard layers only.
No high-level seq2seq trainers or prebuilt pipelines.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Args:
        vocab_size: Size of the source vocabulary.
        embedding_dim: Dimension of token embeddings.
        hidden_dim: Hidden size of the LSTM.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        padding_idx: Index of the <pad> token.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        padding_idx,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        # Bidirectional LSTM — output dim is 2 * hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Project bidirectional hidden/cell to decoder's unidirectional space
        self.hidden_projection = nn.Linear(
            hidden_dim * 2, hidden_dim
        )
        self.cell_projection = nn.Linear(
            hidden_dim * 2, hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, source_ids, source_lengths):
        """
        Args:
            source_ids: (batch_size, src_seq_len) token IDs.
            source_lengths: (batch_size,) actual lengths before padding.

        Returns:
            outputs: (batch_size, src_seq_len, hidden_dim * 2)
            hidden:  (num_layers, batch_size, hidden_dim)
            cell:    (num_layers, batch_size, hidden_dim)
        """
        # (batch_size, src_seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(source_ids))

        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # packed_outputs: (batch_size, src_seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell:   (num_layers * 2, batch_size, hidden_dim)
        packed_outputs, (hidden, cell) = self.lstm(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # Split bidirectional hidden/cell and project to decoder space
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        # We need to combine forward and backward directions per layer
        hidden = self._project_bidirectional_state(hidden)
        cell = self._project_bidirectional_state(cell)

        return outputs, hidden, cell

    def _project_bidirectional_state(self, state):
        """
        Project bidirectional LSTM state to unidirectional decoder state.

        state shape: (num_layers * 2, batch_size, hidden_dim)
        We concatenate forward + backward per layer, then linear project.
        """
        # Reshape: (num_layers, 2, batch_size, hidden_dim)
        state = state.view(
            self.num_layers, 2, state.size(1), self.hidden_dim
        )

        # Concatenate forward and backward: (num_layers, batch_size, hidden_dim * 2)
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)

        # Project to hidden_dim: (num_layers, batch_size, hidden_dim)
        if state is hidden:  # Can't compare like this, use separate logic
            pass
        # Actually let's just handle this in forward
        return state

    def project_hidden(self, state):
        """Project hidden state from bidirectional to unidirectional."""
        state = state.view(
            self.num_layers, 2, state.size(1), self.hidden_dim
        )
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)
        return torch.tanh(self.hidden_projection(state))

    def project_cell(self, state):
        """Project cell state from bidirectional to unidirectional."""
        state = state.view(
            self.num_layers, 2, state.size(1), self.hidden_dim
        )
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)
        return torch.tanh(self.cell_projection(state))

    def forward(self, source_ids, source_lengths):
        """
        Args:
            source_ids: (batch_size, src_seq_len) token IDs.
            source_lengths: (batch_size,) actual lengths before padding.

        Returns:
            outputs: (batch_size, src_seq_len, hidden_dim * 2)
            hidden:  (num_layers, batch_size, hidden_dim)
            cell:    (num_layers, batch_size, hidden_dim)
        """
        # (batch_size, src_seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(source_ids))

        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, (hidden, cell) = self.lstm(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # Project bidirectional states to decoder's unidirectional space
        hidden = self.project_hidden(hidden)
        cell = self.project_cell(cell)

        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention.

    score(s_t, h_i) = v^T * tanh(W_s * s_t + W_h * h_i)
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Encoder outputs are bidirectional: hidden_dim * 2
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim) — top-layer hidden state.
            encoder_outputs: (batch_size, src_seq_len, hidden_dim * 2)
            mask: (batch_size, src_seq_len) — True for real tokens, False for padding.

        Returns:
            context: (batch_size, hidden_dim * 2) — weighted sum of encoder outputs.
            attention_weights: (batch_size, src_seq_len) — softmaxed alignment scores.
        """
        # (batch_size, src_seq_len, hidden_dim)
        wh = self.W_h(encoder_outputs)

        # (batch_size, 1, hidden_dim)
        ws = self.W_s(decoder_hidden).unsqueeze(1)

        # (batch_size, src_seq_len, hidden_dim)
        score_input = torch.tanh(wh + ws)

        # (batch_size, src_seq_len, 1) -> (batch_size, src_seq_len)
        scores = self.v(score_input).squeeze(-1)

        # Mask out padding positions by setting scores to -inf
        scores = scores.masked_fill(~mask, float("-inf"))

        # (batch_size, src_seq_len)
        attention_weights = F.softmax(scores, dim=1)

        # (batch_size, 1, src_seq_len) @ (batch_size, src_seq_len, hidden_dim * 2)
        # = (batch_size, 1, hidden_dim * 2) -> (batch_size, hidden_dim * 2)
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention.

    At each timestep:
      1. Embed the previous token
      2. Concatenate with attention context vector
      3. Pass through LSTM
      4. Project to vocabulary size
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        padding_idx,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        # Input to LSTM: embedding + context vector (hidden_dim * 2)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = BahdanauAttention(hidden_dim)

        # Output projection: LSTM hidden + context vector -> vocab logits
        self.output_projection = nn.Linear(
            hidden_dim + hidden_dim * 2, vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token,
        decoder_hidden,
        decoder_cell,
        encoder_outputs,
        source_mask,
    ):
        """
        Single decoding step.

        Args:
            input_token: (batch_size,) — previous token IDs.
            decoder_hidden: (num_layers, batch_size, hidden_dim)
            decoder_cell:   (num_layers, batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_seq_len, hidden_dim * 2)
            source_mask: (batch_size, src_seq_len)

        Returns:
            prediction: (batch_size, vocab_size) — logits for next token.
            decoder_hidden: (num_layers, batch_size, hidden_dim)
            decoder_cell:   (num_layers, batch_size, hidden_dim)
            attention_weights: (batch_size, src_seq_len)
        """
        # (batch_size, 1, embedding_dim)
        embedded = self.dropout(
            self.embedding(input_token).unsqueeze(1)
        )

        # Attention using the top-layer hidden state
        # (batch_size, hidden_dim)
        top_hidden = decoder_hidden[-1]

        context, attention_weights = self.attention(
            top_hidden, encoder_outputs, source_mask
        )

        # (batch_size, 1, hidden_dim * 2)
        context = context.unsqueeze(1)

        # LSTM input: (batch_size, 1, embedding_dim + hidden_dim * 2)
        lstm_input = torch.cat([embedded, context], dim=2)

        # output: (batch_size, 1, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell:   (num_layers, batch_size, hidden_dim)
        output, (decoder_hidden, decoder_cell) = self.lstm(
            lstm_input, (decoder_hidden, decoder_cell)
        )

        # Concatenate LSTM output with context for prediction
        # (batch_size, hidden_dim + hidden_dim * 2)
        prediction_input = torch.cat(
            [output.squeeze(1), context.squeeze(1)], dim=1
        )

        # (batch_size, vocab_size)
        prediction = self.output_projection(prediction_input)

        return prediction, decoder_hidden, decoder_cell, attention_weights


class Seq2Seq(nn.Module):
    """
    End-to-end sequence-to-sequence model with attention.

    Training: teacher forcing (feed ground-truth previous token).
    Inference: greedy decoding (feed model's own prediction).
    """

    def __init__(
        self,
        encoder,
        decoder,
        source_pad_id,
        target_bos_id,
        target_eos_id,
        device,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_id = source_pad_id
        self.target_bos_id = target_bos_id
        self.target_eos_id = target_eos_id
        self.device = device

    def make_source_mask(self, source_ids):
        """True for real tokens, False for padding."""
        return source_ids != self.source_pad_id

    def forward(self, source_ids, source_lengths, target_ids, teacher_forcing_ratio=0.5):
        """
        Training forward pass with teacher forcing.

        Args:
            source_ids: (batch_size, src_seq_len)
            source_lengths: (batch_size,)
            target_ids: (batch_size, tgt_seq_len) — includes <bos> and <eos>
            teacher_forcing_ratio: Probability of using ground-truth token vs model prediction.

        Returns:
            outputs: (batch_size, tgt_seq_len - 1, vocab_size) — predictions for each timestep
                     (excluding <bos> since we predict starting from it).
        """
        batch_size = source_ids.size(0)
        tgt_seq_len = target_ids.size(1)
        vocab_size = self.decoder.vocab_size

        # Tensor to store all decoder outputs
        outputs = torch.zeros(
            batch_size, tgt_seq_len - 1, vocab_size
        ).to(self.device)

        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(
            source_ids, source_lengths
        )

        source_mask = self.make_source_mask(source_ids)

        # First decoder input is <bos> token
        input_token = target_ids[:, 0]  # (batch_size,)

        for t in range(1, tgt_seq_len):
            prediction, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, source_mask
            )

            # Store prediction
            outputs[:, t - 1, :] = prediction

            # Teacher forcing: use ground truth or model prediction
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                input_token = target_ids[:, t]
            else:
                input_token = prediction.argmax(dim=1)

        return outputs

    def decode(self, source_ids, source_lengths, max_length=60):
        """
        Greedy decoding for inference.

        Args:
            source_ids: (batch_size, src_seq_len) or (src_seq_len,)
            source_lengths: (batch_size,) or scalar
            max_length: Maximum number of tokens to generate.

        Returns:
            predictions: List of lists containing predicted token IDs.
            attention_weights: List of attention weight tensors per example.
        """
        self.eval()

        # Handle single example
        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(0)
            source_lengths = torch.tensor([source_lengths])

        batch_size = source_ids.size(0)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(
                source_ids, source_lengths
            )

            source_mask = self.make_source_mask(source_ids)

            # Start with <bos>
            input_token = torch.full(
                (batch_size,),
                self.target_bos_id,
                dtype=torch.long,
                device=self.device,
            )

            predictions = [[] for _ in range(batch_size)]
            attention_weights = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for t in range(max_length):
                prediction, hidden, cell, attn = self.decoder(
                    input_token, hidden, cell, encoder_outputs, source_mask
                )

                # Get most likely next token
                input_token = prediction.argmax(dim=1)

                for i in range(batch_size):
                    if not finished[i]:
                        token_id = input_token[i].item()
                        predictions[i].append(token_id)
                        attention_weights[i].append(attn[i].cpu())

                        if token_id == self.target_eos_id:
                            finished[i] = True

                # Stop if all sequences finished
                if all(finished):
                    break

        return predictions, attention_weights

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
