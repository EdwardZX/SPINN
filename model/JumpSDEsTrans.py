import torch.nn as nn
import torch
import torch.nn.functional as F


class JumpSDEsTransformer(nn.Module):
    """
    Implements a transformer-based model for processing jump stochastic differential equations (SDEs).
    This model combines a GRU (Gated Recurrent Unit) for sequential data processing and a Transformer
    encoder for capturing long-range dependencies. It also includes feed-forward neural networks (FFNNs)
    for embedding and output transformations, with options for dropout and residual connections.

    Args:
        in_dim (int): Input dimension of the tracks.
        out_dim (int): Output dimension of the tracks.
        h_dim (int): Hidden dimension for the linear embedding and FFNNs.
        hidden_dim (int): Hidden dimension for the GRU.
        layers (int): Number of layers in the GRU.
        encoding_len (int): Length of the encoding sequence.
        default_enc_nn (tuple): Description of the default encoder FFNN layers, e.g., ((50, 'gelu'), (50, 'gelu')).
        d_model (int): Model dimension for the Transformer.
        n_head (int): Number of attention heads in the Transformer.
        layers_enc (int): Number of layers in the Transformer encoder.
        dropout (float): Dropout rate for the GRU and FFNNs.
        device (torch.device): Device to run the model on (CPU or GPU).
        beta_obs (float): Weight for the Total variance (TV) observation loss (default 0.)
        beta_ac (float): Weight for the autocorrelation regularization loss.
        beta_conti (float): Weight for the continuity loss. (default 0.)
        beta_weight (float): Weight for jump term of SDE loss (default 0.5).
        is_var_head (bool): Whether to use a softplus head for Dt.
    """
    def __init__(self, in_dim=2, out_dim=2, h_dim=10, hidden_dim=50, layers=2, encoding_len=50,
                 default_enc_nn=((50, 'gelu'), (50, 'gelu')),
                 d_model=20, n_head=4, layers_enc=3, dropout=0.15,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 beta_obs=0., beta_ac=1, beta_conti=0, beta_weight=0.5,
                 is_var_head=None):

        super(JumpSDEsTransformer, self).__init__()

        # GRU for sequential processing
        self.myrnn = nn.GRU(input_size=h_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            dropout=dropout)
        self.rnn_output = nn.Linear(hidden_dim, h_dim)

        # Store parameters
        self.enc_len = encoding_len
        self.device = device

        # FFNNs for embedding and output
        self.emb = FFNN(input_size=in_dim, output_size=h_dim,
                        nn_desc=default_enc_nn, dropout_rate=dropout)
        self.output = FFNN(input_size=h_dim, output_size=out_dim,
                           nn_desc=default_enc_nn, dropout_rate=dropout)

        # Transformer components
        self.h_enc_embedding = nn.Linear(in_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers_enc)
        self.enc_to_dec = nn.Linear(d_model, h_dim)

        # Loss function weights
        self.beta_obs = beta_obs
        self.beta_ac = beta_ac
        self.beta_conti = beta_conti
        self.beta_weight = beta_weight

        # Optional variance head
        self.is_var_head = is_var_head

    def _generate_square_subsequent_mask(self, sz):
        """Generate attention mask for transformer"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input):
        self.myrnn.flatten_parameters()

        # Transformer encoder
        mask = self._generate_square_subsequent_mask(len(input)).to(self.device)
        seq_emb = self.h_enc_embedding(input)
        enc = self.transformer_encoder(seq_emb, mask)
        hx = self.enc_to_dec(enc)[self.enc_len:]

        # GRU processing
        x = input[self.enc_len:]
        h0 = self.emb(x) + hx
        h = self.rnn_output(self.myrnn(h0)[0])

        # Generate outputs
        if self.is_var_head is not None:
            y0, y1 = F.softplus(self.output(h0)), F.softplus(self.output((h + h0)))
        else:
            y0, y1 = self.output(h0), self.output((h + h0))

        return y0, y1

    def cal_loss(self, y0, y1, x, loss_fn=None, L2=None):
        """
        Calculate SDEs loss function
        """
        X_obs = x[:-1, :]
        X_pre = x[1:, :]
        Y_pre = y1[:-1, :]
        Y_Jump = y0[1:, :]
        Y_obs = y0[:-1, :]
        Y_pre_pre = y1[1:, :]
        X0 = x[0, :]
        Y0 = y0[0, :]
        weight = self.beta_weight
        eps = 1e-16
        v_noise = torch.abs(Y_pre - Y_pre_pre)
        if (loss_fn is not None) and (L2 is None):
            mse = ((1 - weight) * torch.sqrt((X_pre - Y_Jump) ** 2 + eps) +
                   weight * torch.sqrt((Y_pre - Y_Jump) ** 2 + eps)
                   )
            mse += self.beta_obs * v_noise
            mse = loss_fn.lossfun(mse.reshape(mse.shape[1], -1)).sum()  # B x dim
        else:
            mse = ((1 - weight) * torch.sqrt((X_pre - Y_Jump) ** 2 + eps) +
                   weight * torch.sqrt((Y_pre - Y_Jump) ** 2 + eps)
                   ) ** 2
            mse = torch.sum(mse) + self.beta_obs * v_noise.sum()


        mse += self.beta_conti * torch.abs(Y_Jump - Y_obs).sum()

        inner = torch.sum(self.cal_autocorrelation(X_pre - Y_pre), dim=1)
        inner_var = torch.sum(mse) + torch.sum(((1 - weight) * (X0 - Y0)) ** 2)
        inner_ac = torch.abs(inner[1:10])

        recon_loss = torch.sum((X_pre - Y_Jump) ** 2)
        pre_loss = torch.sum((Y_pre - X_pre) ** 2)

        return torch.sum(inner_var) + 0.5 * self.beta_ac * torch.sum(inner_ac), pre_loss, recon_loss, torch.sum(
            inner_ac)

    def cal_loss_filter(self, y0, y1, x):
        """
        Calculate filtered loss
        """
        weight = 0.5
        eps = 1e-16
        mse = ((1 - weight) * torch.sqrt((y1 - x) ** 2 + eps) +
               weight * torch.sqrt((y0 - y1) ** 2 + eps)
               ) ** 2
        inner = torch.sum(self.cal_autocorrelation(x - y1), dim=1)
        inner_var = torch.sum(mse)
        inner_ac = torch.sqrt(inner[1:10] ** 2 + eps)
        recon_loss = torch.sum((y0 - y1) ** 2)
        pre_loss = torch.sum((y1 - x) ** 2)

        return torch.sum(inner_var) + self.beta_ac * torch.sum(inner_ac), pre_loss, recon_loss, torch.sum(inner_ac)

    def cal_autocorrelation(self, x):
        """
        Calculate autocorrelation using FFT
        """
        # Tensor (L,B,D)
        N = x.shape[0]
        X = torch.concat((x, torch.zeros(N - 1, x.shape[1], x.shape[2]).to(self.device)), dim=0)
        fx = torch.fft.fft(X, dim=0)
        ac = torch.fft.ifft(fx * torch.conj(fx), dim=0).real
        return ac[:N]

    def cal_loss_sigma(self, y0, y1, x):
        """
        Calculate SDEs loss function with weight modifying
        """
        X_obs = x[:-1, :]
        X_pre = x[1:, :]
        Y_pre = y1[:-1, :]
        Y_Jump = y0[1:, :]
        Y_obs = y0[:-1, :]
        Y_pre_pre = y1[1:, :]

        X0 = x[0, :]
        Y0 = y0[0, :]
        weight = self.beta_weight

        eps = 1e-6

        # dX = torch.abs(X_pre - X_obs)
        # dX[dX<1.] = 1.
        # dX = dX ** (-0.5)

        ## surpress the outliers
        dX = torch.abs(Y_pre.detach() - Y_obs.detach())
        dX[dX < 1.] = 1.
        dX = dX ** (-0.5)

        v_noise = torch.abs(Y_pre - Y_pre_pre)
        # inner_obs = self.beta_obs * v_noise.sum()

        # eps = 1e-6

        mse = ((1 - weight) * torch.sqrt((X_pre - Y_Jump) ** 2 + eps)
               + weight * torch.sqrt((Y_pre - Y_Jump) ** 2 + eps)
               ) ** 2

        ##
        mse = mse * dX

        mse += self.beta_obs * v_noise

        # mse = loss_fn.lossfun(mse.reshape(mse.shape[1], -1)).sum()

        # var

        # inner = torch.sum(self.cal_autocorrelation(X_pre - Y_pre), dim=1)
        inner = torch.sum(self.cal_autocorrelation(dX * (X_pre - Y_pre)), dim=1)
        inner_var = torch.sum(mse) + torch.sum(((1 - weight) * (X0 - Y0)) ** 2)
        inner_ac = torch.sqrt(inner[1:10] ** 2 + eps)
        # inner_ac = torch.sqrt(inner_jump[1:10] ** 2 + eps) + torch.sqrt(inner_pred[1:10] ** 2 + eps)
        recon_loss = torch.sum((X_pre - Y_Jump) ** 2)
        pre_loss = torch.sum((X_pre - Y_pre) ** 2)

        ## Gaussian_nll_loss
        loss_mle = F.gaussian_nll_loss(torch.sqrt(X_pre + eps),
                                       torch.zeros_like(X_pre).to(self.device),
                                       torch.abs(Y_pre), reduction='sum')

        return torch.sum(inner_var) + 0.5 * self.beta_ac * torch.sum(inner_ac), pre_loss, recon_loss, torch.sum(
            loss_mle)

        # return  torch.sum(loss_bmle), pre_loss, recon_loss, torch.sum(loss_mle)

    def beta_nll_loss(self, variance, target, beta=0.5):
        """Compute beta-NLL loss

        :param mean: Predicted mean of shape B x D
        :param variance: Predicted variance of shape B x D
        :param target: Target of shape B x D
        :param beta: Parameter from range [0, 1] controlling relative
            weighting between data points, where `0` corresponds to
            high weight on low error points and `1` to an equal weighting.
        :returns: Loss per batch element of shape B
        """
        # eps = 1e-06

        # Clamp for stability
        # variance = variance.clone()
        # with torch.no_grad():
        #     variance.clamp_(min=eps)

        loss = 0.5 * (target ** 2 / variance + variance.log())

        if beta > 0:
            loss = loss * (variance.detach() ** beta)

        return loss.sum(axis=-1)


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias=True):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    nonlinears = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'gelu': torch.nn.GELU
    }

    if nn_desc is None:
        layers = [torch.nn.Linear(input_size, output_size, bias=bias)]
    else:
        layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]
        if len(nn_desc) > 1:
            for i in range(len(nn_desc) - 1):
                layers.append(nonlinears[nn_desc[i][1]]())
                layers.append(torch.nn.Dropout(p=dropout_rate))
                layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i + 1][0],
                                              bias=bias))
        layers.append(nonlinears[nn_desc[-1][1]]())
        layers.append(torch.nn.Dropout(p=dropout_rate))
        layers.append(torch.nn.Linear(nn_desc[-1][0], output_size, bias=bias))
    return torch.nn.Sequential(*layers)


class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with gelu applied to inputs and the
    option to use a residual NN version (then the output size needs to be a
    multiple of the input size or vice versa)
    """

    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=True, masked=False):
        super().__init__()

        # create feed-forward NN
        in_size = input_size
        if masked:
            in_size = 2 * input_size
        self.masked = masked
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=output_size,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias
        )

        if residual:
            # print('use residual network: input_size={}, output_size={}'.format(
            #     input_size, output_size))
            if input_size <= output_size:
                if output_size % input_size == 0:
                    self.case = 1
                    self.mult = int(output_size / input_size)
                else:
                    raise ValueError('for residual: output_size needs to be '
                                     'multiple of input_size')

            if input_size > output_size:
                if input_size % output_size == 0:
                    self.case = 2
                    self.mult = int(input_size / output_size)
                else:
                    raise ValueError('for residual: input_size needs to be '
                                     'multiple of output_size')
        else:
            self.case = 0

    def forward(self, nn_input, mask=None):
        if self.masked:
            assert mask is not None
            # out = self.ffnn(torch.cat((F.gelu(nn_input), mask), 1))
            out = self.ffnn(torch.cat((nn_input, mask), 1))
        else:
            # out = self.ffnn(F.gelu(nn_input))
            out = self.ffnn(nn_input)

        if self.case == 0:
            return out
        elif self.case == 1:
            # a = nn_input.repeat(1,1,self.mult)
            identity = nn_input.repeat(1, 1, self.mult)
            return identity + out
        elif self.case == 2:
            identity = torch.mean(torch.stack(nn_input.chunk(self.mult, dim=-1)),
                                  dim=0)
            return identity + out


class RawGRU(nn.Module):
    """
    GRU-based generator model that combines transformer encoder with GRU decoder

    Args:
        myrnn: Base RNN model to copy components from
        h_dim: Input dimension
        hidden_dim: Hidden layer dimension
        layers: Number of GRU layers
        generator_num: Number of generation steps
    """
    def __init__(self, myrnn, h_dim=10, hidden_dim=50, layers=2, generator_num=30):
        super(RawGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.h_dim = h_dim
        self.emb = myrnn.emb
        self.rnn_output = myrnn.rnn_output
        self.output = myrnn.output
        self.generator_num = generator_num
        self.layers = layers
        self.enc_len = myrnn.enc_len
        self.myrnnRaw = myrnn
        w = myrnn.myrnn
        self.rnn_cell = RawGRUCell(w, self.h_dim, self.hidden_dim, self.layers)

    def forward(self, x, sigma, jumping=None):
        # x0 = batch x dimension
        # x: (head + seq) x batch x D; sigma: seq x batch x D
        input = x[:self.enc_len]
        x_0 = x[self.enc_len]
        h = None
        y0 = []
        y1 = []
        x0 = []
        for i in range(self.generator_num):
            raw_x_0 = x_0
            input = torch.cat([input, x_0.unsqueeze(dim=0)])
            x0.append(raw_x_0)
            mask = self.myrnnRaw._generate_square_subsequent_mask(len(input)).to(self.myrnnRaw.device)
            seq_emb = self.myrnnRaw.h_enc_embedding(input)
            enc = self.myrnnRaw.transformer_encoder(seq_emb, mask)
            hx = self.myrnnRaw.enc_to_dec(enc)[self.enc_len + i]

            h0 = self.emb(x_0).squeeze() + hx
            dh, h = self.rnn_cell(h0, h)
            y = self.output(self.rnn_output(dh) + h0)
            y0.append(self.output(h0))
            y1.append(y)

            if jumping is not None:
                x_0 = y + sigma[i] * torch.randn_like(x_0) + jumping[i] * torch.ones_like(x_0)
            else:
                # dx = sigma[i] * torch.randn_like(x_0)
                x_0 = y + sigma[i] * torch.randn_like(x_0)  # adding perturb
            # x_0 = y + alpha * torch.randn_like(x_0)  # adding perturb

            # input = torch.cat([input, x_0])
        return torch.stack(y0, dim=0), torch.stack(y1, dim=0), torch.stack(x0, dim=0)


class RawGRUCell(nn.Module):
    """
    Custom GRU cell that initializes weights from a pre-trained RNN

    Args:
        myrnn: Pre-trained RNN model to copy weights from
        h_dim: Input dimension
        hidden_dim: Hidden layer dimension
        layers: Number of GRU layers
    """
    def __init__(self, myrnn, h_dim=10, hidden_dim=50, layers=2):
        super(RawGRUCell, self).__init__()
        w = myrnn
        self.rnn_cell_list = nn.ModuleList()
        self.layers = layers

        # Create GRU cells for each layer
        for i in range(self.layers):
            input_size = h_dim if i == 0 else hidden_dim
            self.rnn_cell_list.append(self.initGell(nn.GRUCell(input_size=input_size,
                                                               hidden_size=hidden_dim), i, w))

    def initGell(self, rnncell, layer, RNNbase):
        """
        Initialize GRU cell weights from pre-trained model

        Args:
            rnncell: GRU cell to initialize
            layer: Layer number
            RNNbase: Pre-trained RNN to copy weights from

        Returns:
            Initialized GRU cell
        """
        rnncell.state_dict()['weight_ih'][:] = getattr(RNNbase, 'weight_ih_l' + str(layer))
        rnncell.state_dict()['weight_hh'][:] = getattr(RNNbase, 'weight_hh_l' + str(layer))

        rnncell.state_dict()['bias_ih'][:] = getattr(RNNbase, 'bias_ih_l' + str(layer))
        rnncell.state_dict()['bias_hh'][:] = getattr(RNNbase, 'bias_hh_l' + str(layer))

        return rnncell

    def forward(self, x, h=None):
        h_output = []

        for i in range(self.layers):
            rnn_cell = self.rnn_cell_list[i]
            if h is not None:
                x = rnn_cell(x, h[i])
            else:
                x = rnn_cell(x)

            h_output.append(x)
        return x, h_output
