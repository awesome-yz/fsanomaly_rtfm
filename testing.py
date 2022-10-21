import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
from datasets.get_features import GetFeatures
import scipy.stats
import datasets
import models
import utils
import yaml
import argparse
import torch


def mean_confidence_interval(data, confidence=0.95):
    # import pdb; pdb.set_trace
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def main(config):
    #import pdb; pdb.set_trace()
    n_way = config['n_way']
    n_query = config['n_query']
    n_shot = config['n_shot']

    abnorm_features = GetFeatures(config['root_path'], 'Abnormal')
    norm_features = GetFeatures(config['root_path'], 'Normal')

    test_dataset_args = {'abnorm_feats': abnorm_features.data, 'norm_feats': norm_features.data, 'n_cls': n_way,\
         'n_sprt': n_shot, 'n_qry': n_query, 'n_batch': 1}
    test_dataset = datasets.make(config['test_dataset'], **test_dataset_args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # import pdb; pdb.set_trace()
    if config.get('load_model') is not None:
        model = models.load(torch.load(config['load_model']))
    else:
        # model_args = {'rtfm_args': config['rtfm_args'], 'n_way': n_way, 'n_shot': n_shot, 'n_query': n_query}
        model = models.make(config['model'], **config['model_args'])

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    test_epochs = config['epochs']
    aves_keys = ['va']
    aves = {k: utils.Averager() for k in aves_keys}
    va_lst = []

    if config.get('lmd') is not None:
        for lmd in config['lmd']:
            print('lambda value {}\n'.format(lmd))
            for epoch in range(1, test_epochs + 1):
                for data in test_loader:
                    with torch.no_grad():
                        predictions, labels= model(data, n_way, n_shot, n_query, lmd)
                        # loss = F.nll_loss(predictions, labels, reduction="sum") + intra_loss
                        #import pdb; pdb.set_trace()
                        acc = torch.mean((predictions==labels).float())
                        aves['va'].add(acc.item(), len(data))
                        va_lst.append(acc.item())

            #import pdb; pdb.set_trace()
            print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
                epoch, aves['va'].item() * 100, mean_confidence_interval(va_lst) * 100))

    else:
        for epoch in range(1, test_epochs + 1):
            for data in test_loader:
                with torch.no_grad():
                    predictions, labels, _ = model(data)
                    # loss = F.nll_loss(predictions, labels, reduction="sum") + intra_loss
                    acc = utils.compute_acc(predictions, labels)
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc.item())

        # import pdb; pdb.set_trace()
        print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, aves['va'].item() * 100, mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_var_model_mini.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    utils.set_gpu(args.gpu)
    main(config)
