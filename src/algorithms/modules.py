import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_out_shape(in_shape, layers, attn=False):
	x = torch.randn(*in_shape).unsqueeze(0)
	if attn:
		return layers(x, x, x).squeeze(0).shape
	else:
		return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def orthogonal_init(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class NormalizeImg(nn.Module):
	def __init__(self, mean_zero=False):
		super().__init__()
		self.mean_zero = mean_zero

	def forward(self, x):
		if self.mean_zero:
			return x/255. - 0.5
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class Identity(nn.Module):
	def __init__(self, obs_shape=None, out_dim=None):
		super().__init__()
		self.out_shape = obs_shape
		self.out_dim = out_dim
	
	def forward(self, x):
		return x

#
class RandomShiftsAug(nn.Module):
	def __init__(self, pad):
		super().__init__()
		self.pad = pad

	def forward(self, x):
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps,
								1.0 - eps,
								h + 2 * self.pad,
								device=x.device,
								dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

		shift = torch.randint(0,
							  2 * self.pad + 1,
							  size=(n, 1, 1, 2),
							  device=x.device,
							  dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)

		grid = base_grid + shift
		return F.grid_sample(x,
							 grid,
							 padding_mode='zeros',
							 align_corners=False)


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32, mean_zero=False):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters
		self.layers = [NormalizeImg(mean_zero), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(orthogonal_init)

	def forward(self, x):
		return self.layers(x)


class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32, flatten=True):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		if flatten:
			self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(orthogonal_init)

	def forward(self, x):
		return self.layers(x)


class Integrator(nn.Module):
	def __init__(self, in_shape_1, in_shape_2, num_filters=32, concatenate=True):
		super().__init__()
		self.relu = nn.ReLU()
		if concatenate:
			self.conv1 = nn.Conv2d(in_shape_1[0]+in_shape_2[0], num_filters, (1,1))
		else:
			self.conv1 = nn.Conv2d(in_shape_1[0], num_filters, (1,1))
		self.apply(orthogonal_init)

	def forward(self, x):
		x = self.conv1(self.relu(x))
		return x


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection, attention=None):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.attention = attention
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		x = self.projection(x)
		return x
		
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Actor(nn.Module):
	def __init__(self, out_dim, projection_dim, state_shape, action_shape, hidden_dim, hidden_dim_state, log_std_min, log_std_max):
		super().__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.trunk = nn.Sequential(nn.Linear(out_dim, projection_dim),
								   nn.LayerNorm(projection_dim), nn.Tanh())

		self.layers = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim), nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)

		if state_shape:
			self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
			                                   nn.ReLU(inplace=True),
			                                   nn.Linear(hidden_dim_state, projection_dim),
			                                   nn.LayerNorm(projection_dim), nn.Tanh())
		else:
		    self.state_encoder = None

		self.apply(orthogonal_init)

	def forward(self, x, state, compute_pi=True, compute_log_pi=True):
		x = self.trunk(x)

		if self.state_encoder:
		    x = x + self.state_encoder(state)

		mu, log_std = self.layers(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class Critic(nn.Module):
	def __init__(self, out_dim, projection_dim, state_shape, action_shape, hidden_dim, hidden_dim_state):
		super().__init__()
		self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
								   nn.LayerNorm(projection_dim), nn.Tanh())

		if state_shape:
		    self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
		                                       nn.ReLU(inplace=True),
		                                       nn.Linear(hidden_dim_state, projection_dim),
		                                       nn.LayerNorm(projection_dim), nn.Tanh())
		else:
		    self.state_encoder = None
		
		self.Q1 = nn.Sequential(
			nn.Linear(projection_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
		self.Q2 = nn.Sequential(
			nn.Linear(projection_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
		self.apply(orthogonal_init)

	def forward(self, obs, state, action):
		obs = self.projection(obs)

		if self.state_encoder:
			obs = obs + self.state_encoder(state)

		h = torch.cat([obs, action], dim=-1)
		return self.Q1(h), self.Q2(h)
