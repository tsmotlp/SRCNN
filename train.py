import argparse
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import os
import cv2
import numpy as np
import random
from torchvision import transforms
from skimage import io as sio
from models import SRCNN
from dataset import get_dataset
from vis_tools import Visualizer

parser = argparse.ArgumentParser (description='pytorch SRCNN')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument ('--num_epochs', type=int, default=500, help='number of training epochs')
parser.add_argument ('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument ('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument ('--resume', type=str, default='', help='path to network checkpoint')
parser.add_argument ('--start_epoch', type=int, default=1, help='restart epoch number for training')
parser.add_argument ('--threads', type=int, default=1, help='number of threads')
parser.add_argument ('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument ('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument ('--step', type=int, default=200, help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10')
parser.add_argument ('--pretrained', type=str, default='', help='path to network parameters')
parser.add_argument ('--scale_factor', type=int, default=2, help='scale factor')
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument ('--LR_train_dir', type=str, default='./dataset/train/', help='LR image path to training data directory')
parser.add_argument ('--HR_train_dir', type=str, default='./dataset/train_label/', help='HR image path to training data directory')
parser.add_argument ('--LR_test_dir', type=str, default='./dataset/test/', help='LR image path to testing data directory')
parser.add_argument ('--HR_test_dir', type=str, default='./dataset/test_label/', help='HR image path to testing data directory')
parser.add_argument ('--train_interval', type=int, default=50, help='interval for training to save image')
parser.add_argument ('--test_interval', type=int, default=10, help='interval for testing to save image')
opt = parser.parse_args ()

# 打印定义的变量
# print(opt)

# ...

seed = random.randint (1, 10000)
print ("Random Seed: ", seed)
torch.manual_seed (seed)
if opt.cuda:
	torch.cuda.manual_seed (seed)


# 构建网络
print('==>building network...')
network = SRCNN(1)

# loss函数
loss_func = torch.nn.MSELoss()

# 设置GPU
if opt.cuda and not torch.cuda.is_available():  # 检查是否有GPU
	raise Exception('No GPU found, please run without --cuda')
print("===> Setting GPU")
if opt.cuda:
	print('cuda_mode:', opt.cuda)
	network = network.cuda()
	loss_func = loss_func.cuda()

# 设置优化器函数
print("===> Setting Optimizer")
optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

# 可视化
train_vis = Visualizer (env='training')


# 训练
def train(train_dataloader, network, optimizer, loss_func, save_img_dir):
	print('==>Training...')
	for epoch in range(opt.start_epoch, opt.num_epochs + 1):
		# scheduler.step(epoch)
		train_process(train_dataloader, network, optimizer, loss_func, save_img_dir, epoch, epochs=opt.num_epochs, interval=opt.train_interval)
		save_checkpoint(network, epoch)


# 测试
def test(test_dataloader, network, save_img_dir):
	print ('==>Testing...')
	test_process (test_dataloader, network, save_img_dir)


# 每个epoch的训练程序
def train_process(dataloader, network, optimizer, loss_func, save_img_dir, epoch=1, epochs=1, interval=100):
	lr = adjust_learning_rate (epoch - 1)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
		print ("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
	for iteration, (inputs, labels) in enumerate (dataloader):
		inputs = Variable (inputs)  # 输入数据
		labels = Variable (labels)  # label

		if opt.cuda:
			inputs = inputs.cuda ()
			labels = labels.cuda ()
		# -----------------------------------------------
		# training discriminator
		# ------------------------------------------------
		optimizer.zero_grad ()
		gen_hr = network (inputs)  # 网络输出

		loss = loss_func(gen_hr, labels)

		loss.backward()
		optimizer.step()
		psnr = PSNR(gen_hr, labels)

		idx = np.random.choice (opt.batch_size)
		train_vis.plot ('loss', loss.item ())
		train_vis.plot ('psnr', psnr.item ())
		train_vis.img ('LR image', inputs[idx].cpu ().detach ().numpy ())
		train_vis.img ('HR image', gen_hr[idx].mul (255).cpu ().detach ().numpy ())
		train_vis.img ('GT image', labels[idx].cpu ().detach ().numpy ())

		# 打印结果：
		# print('fake_out:{} real_out:{} L1_loss:{}'.format (fake_out, real_out, L1_loss (fake_imgs, real_imgs),edge_loss (fake_imgs, real_imgs)))
		print('epoch:[{}/{}] batch:[{}/{}] loss:{:.10f} psnr:{:.10f}'.format(epoch, epochs, iteration, len(dataloader), loss.data[0], psnr))

		if iteration % interval == 0:
			sr = tensor_to_np(gen_hr[idx])
			gt = tensor_to_np(labels[idx])
			fig = cv2.hconcat((sr, gt))
			save_train_img(fig, save_img_dir, 'fig', epoch, iteration)

# 测试程序
def test_process(test_dataloader, network, save_img_dir):
	for idx, (inputs, labels) in enumerate(test_dataloader):
		inputs = Variable(inputs)
		labels = Variable(labels)
		if opt.cuda:
			inputs = inputs.cuda()
			labels = labels.cuda()
		prediction = network(inputs)
		psnr = PSNR(prediction, labels)
		i = np.random.choice (opt.batch_size)
		sr = tensor_to_np (prediction[i])
		gt = tensor_to_np (labels[i])
		fig = cv2.hconcat ((sr, gt))
		save_test_img (fig, save_img_dir, 'test_fig', idx)
		print('batch{} ==> psnr:{}'.format(idx, psnr))

# 设计自适应的学习率
def adjust_learning_rate(epoch):
	lr = opt.lr * (0.5 ** (epoch // opt.step))
	return lr


loader = transforms.Compose ([transforms.ToTensor ()])
unloader = transforms.ToPILImage ()


def tensor_to_np(tensor):
	img = tensor.mul (255).byte ()
	img = img.cpu ().numpy ().squeeze ()
	return img


def save_train_img(image, image_dir, img_name, epoch, iteration):
	if not os.path.exists (image_dir):
		os.mkdir (image_dir)
	image_path = os.path.join (image_dir, img_name + '{}_{}.png'.format (epoch, iteration))
	sio.imsave (image_path, image)

def save_test_img(image, image_dir, img_name, iteration):
	if not os.path.exists (image_dir):
		os.mkdir (image_dir)
	image_path = os.path.join (image_dir, img_name + '{}.png'.format (iteration))
	sio.imsave (image_path, image)

def PSNR(pred, gt):
	pred = pred.cpu ().detach ()
	gt = gt.cpu ().detach ()
	pred = pred.clamp (0, 1)
	diff = pred - gt
	mse = np.mean (diff.numpy () ** 2)
	if mse == 0:
		return 100
	return 10 * np.log10 (1.0 / mse)


def save_checkpoint(network, epoch):
	model_folder = "model_para/"
	param_path = model_folder + "param_epoch{}.pkl".format(epoch)
	state = {"epoch": epoch, "model": network}
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	torch.save(state, param_path)
	print("Checkpoint saved to {}".format(param_path))


# 判断网络是否已经训练过或者已经训练完成
if opt.pretrained:  # 训练完成,进行测试
	# 加载测试数据进行测试
	print('==>loading test data...')
	test_dataset = get_dataset(opt.LR_test_dir, opt.HR_test_dir, opt.scale_factor)
	test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
	if os.path.isfile(opt.pretrained):
		print('==> loading model {}'.format(opt.pretrained))
		weights = torch.load(opt.pretrained)
		network.load_state_dict(weights['model'].state_dict())
		# 进行测试
		test(test_dataloader, network, './test_res')
	else:
		print('==> no network model found at {}'.format(opt.pretrained))
else:  # 未训练完成，需要进行训练
	# 加载训练数据
	print('==>loading training data...')
	train_dataset = get_dataset(opt.LR_train_dir, opt.HR_train_dir, opt.scale_factor)
	train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
	if opt.resume:  # 部分训练，需要重新开始训练
		if os.path.isfile(opt.resume):
			checkpoint = torch.load(opt.resume)
			opt.start_epoch = checkpoint['epoch'] + 1
			print('==>start training at epoch {}'.format(opt.start_epoch))
			network.load_state_dict(checkpoint['model'].state_dict())
			print("===> resume Training...")
			train(train_dataloader, network, optimizer, loss_func, './train_res')
		else:
			print('==> cannot start training at epoch {}'.format(opt.start_epoch))
	else:
		train(train_dataloader, network, optimizer, loss_func, './train_res')
