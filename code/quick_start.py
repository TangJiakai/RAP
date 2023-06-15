import argparse
import logging
import sys

sys.path.append('..')

from utils import get_model, init_data_path, init_seed, init_logger, init_device, output_config
from dataset import data_preparation
from trainer import Trainer
from change_model_config import change_default_config
from log2excel import log_exper_results


def arg_config():
    parser = argparse.ArgumentParser(description='RAP')
    parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--train_bs', default=4096, help='training batch size', type=int)
    parser.add_argument('--test_bs', default=4096, help='testing batch size', type=int)
    parser.add_argument('--embedding_size', default=64, help='embedding size', type=int)
    parser.add_argument('--weight_decay', default=0.000001, help='weight decay for optimizer', type=float)
    parser.add_argument('--optimizer_type', default='adam', choices=['adam', 'sgd'], type=str)
    
    parser.add_argument('--model', default='RAP', type=str)
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--num_workers', default=10, help='number of workers to load data', type=int)
    parser.add_argument('--eval_step', default=10, help='eval interval step num', type=int)
    parser.add_argument('--stopping_step', default=10, help='stopping step num', type=int)
    parser.add_argument('--dataset', default='ml-1M', choices=['ml-100K', 'ml-1M', 'amazon-toys', 'yelp'], help="pretrain data type: ['ml-100K', 'ml-1M', 'amazon-toys', 'yelp']", type=str)
    parser.add_argument('--use_tensorboard', action='store_true', help='choose to use tensorboard')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--use_gpu', action='store_true', help='choose weather to use GPU')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--id', type=int, help='id for specifying current event for various directory')
    parser.add_argument('--log', action='store_true',  help='choose to log info to file')
    parser.add_argument('--saved', action='store_true', help='choose to save best model')
    parser.add_argument('--resume_id', help='resume model checkpoint id', type=int)
    parser.add_argument('--test', action='store_true', help='only test without training')

    parser.add_argument('--noise_ratio', default=0., help='noise interaction ratio', type=float)

    # RAP
    parser.add_argument('--eps', default=0.1, help='noise scale of RAP model', type=float)
    parser.add_argument('--RAP_layer_num', default=1, help='layer number of RAP model', type=int)
    parser.add_argument('--beta', default=0.0, help='hard prune threshold of RAP model', type=float)
    parser.add_argument('--adv_loss_weight', default=1.0, help='adv_loss_weight of RAP model', type=float)

    arg = parser.parse_args()
    return vars(arg)

def main():
    config = arg_config()

    init_data_path(config)
    init_logger(config)
    init_device(config)
    init_seed(config['seed'])

    logger = logging.getLogger()
    logger.info(output_config(config))

    train_dataloader, valid_dataloader, test_dataloader = data_preparation(config)
    model = get_model(config['model'])(config, train_dataloader.dataset).to(config['device'])
    logger.info(model)
    
    trainer = Trainer(config, model)

    if config['resume_id']:
        trainer.resume_checkpoint(config['resume_file'])

    if not config['test']:
        trainer.fit(train_dataloader, valid_dataloader)

    init_seed(config['seed'])
    _, test_result = trainer.evaluate(test_dataloader, load_best_model=True)
    test_result['id'] = config['id']
    return test_result

if __name__ == '__main__':
    main()
