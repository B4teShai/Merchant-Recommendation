"""
SelfGNN Flow: set seed and device -> DataHandler.LoadData() -> Recommender(handler, device).run().
run() trains for args.epoch with periodic test; saves best by NDCG.
"""
from Params import args
from Utils.compat import set_seed, get_device
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import DataHandler
from model import Recommender

if __name__ == '__main__':
	logger.saveDefault = True
	set_seed()
	device = get_device()
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')
	recom = Recommender(handler, device)
	recom.run()