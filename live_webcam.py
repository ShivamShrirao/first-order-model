import time
from tqdm import tqdm
import torch
import numpy as np
import imageio
from skimage.transform import resize
from argparse import ArgumentParser
from threading import Thread
import cv2

from demo import load_checkpoints
from animate import normalize_kp


parser = ArgumentParser()
parser.add_argument("--source_image", default='../../Downloads/first_order/02.png', help="path to source image")
parser.add_argument("--source_cam", default=0, help="source webcam integer")
parser.add_argument("--fake_cam_path", default=None, help="fake webcam path")
parser.add_argument("--fps", default=30, help="output schedule_frame fps")
parser.add_argument("--checkpoint_path", default='../../Downloads/vox-cpk.pth.tar', help="checkpoint path")
opt = parser.parse_args()

cpu=False
fps = int(opt.fps)
source_image = imageio.imread(opt.source_image)
source_image = (resize(source_image, (256, 256))[..., :3]).astype(np.float32)

def make_animate_live(driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
	with torch.no_grad():
		driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
		driving_frame = driving[:, :, 0]
		if not cpu:
			driving_frame = driving_frame.cuda()
		kp_driving = kp_detector(driving_frame)
		kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
							   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
							   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
		out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

		pred = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
	return pred

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path='../../Downloads/vox-cpk.pth.tar')

if opt.fake_cam_path:
	import pyfakewebcam
	fake_camera = pyfakewebcam.FakeWebcam(opt.fake_cam_path, 256, 256)
cap = cv2.VideoCapture(int(opt.source_cam))

ret, frame = cap.read()
slcs = (slice(120, -140, None), slice(455, -455, None))
frame = frame[slcs]
cv2.imshow('frame', frame)
frame = cv2.resize(frame, (256, 256))
img = frame.astype(np.float32)/255

with torch.no_grad():
	source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	if not cpu:
		source = source.cuda()
	driving = torch.tensor(np.array([img])[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
	kp_driving_initial = kp_detector(driving[:, :, 0])
	kp_source = kp_detector(source)

out = source_image
def thread_schedule_fake():
	while True:
		fake_camera.schedule_frame(out)
		time.sleep(1/fps)

if opt.fake_cam_path:
	t = Thread(target=thread_schedule_fake)
	t.setDaemon(True)
	t.start()

while True:
	ret, frame = cap.read()
	frame = frame[slcs]
	cv2.imshow('frame', frame)
	frame = cv2.resize(frame, (256, 256))
	img = frame.astype(np.float32)/255
	pred = make_animate_live([img], generator, kp_detector)

	out = cv2.flip((pred*255).astype(np.uint8), 1)
	cv2.imshow('out', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()