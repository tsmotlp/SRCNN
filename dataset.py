from torchvision import transforms
from PIL import Image
from torch.utils import data as data
import os


def is_image_file(filename):
	return any (filename.endswith (extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])


def load_img(filepath):
	y = Image.open (filepath).convert ('L')
	return y


def x2_bicubic(LR_img):
	h, w = LR_img.size
	return transforms.Resize((h * 2, w * 2), Image.BICUBIC)(LR_img)


def x4_bicubic(LR_img):
	h, w = LR_img.size
	return transforms.Resize((h * 4, w * 4), Image.BICUBIC)(LR_img)


def x8_bicubic(LR_img):
	h, w = LR_img.size
	return transforms.Resize((h * 8, w * 8), Image.BICUBIC)(LR_img)


def resize(img, scale_factor=2):
	h, w = img.size
	return transforms.Resize((w // scale_factor, h // scale_factor))(img)


def to_tensor(img):
	return transforms.ToTensor()(img)


def interpolation(img, scale_factor):
	if scale_factor == 2:
		img = resize(img, 2)
		return x2_bicubic(img)
	if scale_factor == 4:
		img = resize(img, 4)
		return x4_bicubic(img)
	if scale_factor == 8:
		img = resize(img, 8)
		return x8_bicubic(img)


class DatasetFromFolder (data.Dataset):
	def __init__(self, LR_image_dir, HR_image_dir, scale_factor):
		super (DatasetFromFolder, self).__init__ ()
		self.scale_factor = scale_factor
		self.LR_image_filenames = sorted (
			[os.path.join (LR_image_dir, x) for x in os.listdir (LR_image_dir) if is_image_file (x)])
		self.HR_image_filenames = sorted (
			[os.path.join (HR_image_dir, y) for y in os.listdir (HR_image_dir) if is_image_file (y)])


	def __getitem__(self, index):
		inputs = load_img (self.LR_image_filenames[index])
		labels = load_img (self.HR_image_filenames[index])
		inputs = interpolation(inputs, self.scale_factor)
		HR_images = to_tensor (labels)
		LR_images = to_tensor (inputs)
		return LR_images, HR_images

	def __len__(self):
		return len (self.LR_image_filenames)


def get_dataset(LR_image_dir=None, HR_image_dir=None, scale_factor=2):
	return DatasetFromFolder (LR_image_dir, HR_image_dir, scale_factor)
