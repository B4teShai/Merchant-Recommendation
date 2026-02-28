"""
SelfGNN Flow: set seed and device -> DataHandler.LoadData() -> Recommender(handler, device).run().
run() trains for args.epoch with periodic test; saves best by NDCG.
"""
import torch
from Params import args
from Utils.compat import set_seed, get_device
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import DataHandler
from model import Recommender

if __name__ == '__main__':
	logger.saveDefault = True
	set_seed()
	device = get_device(device_override=args.device)
	print("Device:", device)
	if device.type == "cuda":
		print("GPU:", torch.cuda.get_device_name(device))
		torch.backends.cudnn.benchmark = True  # faster conv/attention on fixed input sizes
		# TF32 on Ampere+ (e.g. RTX 5090) for faster matmuls
		if hasattr(torch.backends.cuda, "matmul"):
			torch.backends.cuda.matmul.allow_tf32 = True
		if hasattr(torch.backends.cudnn, "allow_tf32"):
			torch.backends.cudnn.allow_tf32 = True
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	args.graphNum = len(handler.subMat)  # use data's time intervals (e.g. 5 for amazon, 8 for yelp)
	log('Load Data')
	recom = Recommender(handler, device)
	recom.run()