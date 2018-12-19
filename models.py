import torch
import torch.nn as nn


class SRCNN(torch.nn.Module):
	def __init__(self, num_channels=1):
		super(SRCNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
		self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
		self.lrelu2 = nn.LeakyReLU (negative_slope=0.2, inplace=True)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=5, stride=1, padding=2, bias=True)
		for m in self._modules:
			if isinstance (m, nn.Conv2d):
				m.weight.data.normal_ (0, 0.01)
				m.bias.data.zero_ ()
			if isinstance (m, nn.ConvTranspose2d):
				m.weight.data.normal_ (0, 0.01)
				m.bias.data.zero_ ()

	def forward(self, x):
		x = self.lrelu1(self.conv1(x))
		x = self.lrelu2(self.conv2(x))
		out = self.conv3(x)
		return out
