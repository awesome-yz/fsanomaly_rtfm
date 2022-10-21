import argparse
import yaml
import torch
import torch.nn.functional as F
import utils
import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from datasets.samplers import CategoriesSampler
from datasets.get_features import GetFeatures
import models
import os
from sklearn.model_selection import KFold
import numpy
from collections import OrderedDict

tb = SummaryWriter()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def rename_params(params):
    new_dict = OrderedDict()
    for param_name in params.keys():
        new_name =  'rtfm.' + param_name
        new_dict[new_name] = params[param_name]
    return new_dict

def main(config):
    

    n_way = config['n_way']
    n_query = config['n_query']
    n_shot = config['n_shot']

    
    ### training and validation dataset #####
    # import pdb; pdb.set_trace()

    kfold = KFold(n_splits=config['k_folds'], shuffle=True)
    abnorm_features = GetFeatures(config['root_path'], 'Abnormal')
    norm_features = GetFeatures(config['root_path'], 'Normal')

    # import pdb; pdb.set_trace()
    for fold, (norm_train, norm_val) in enumerate(kfold.split(norm_features)) : # using normal data indexes only
        # for _, (norm_train, norm_val) in enumerate(kfold.split(norm_features)):

        ab_train, ab_val = numpy.array(abnorm_features.data)[norm_train], numpy.array(abnorm_features.data)[norm_val]
        nor_train, nor_val = numpy.array(norm_features.data)[norm_train], numpy.array(norm_features.data)[norm_val]
        
        train_dataset_args = {'abnorm_feats': ab_train, 'norm_feats': nor_train, 'n_cls': n_way,\
                                'n_sprt': n_shot, 'n_qry': n_query, 'n_batch': config['train_batches']}
        train_dataset = datasets.make(config['train_dataset'], **train_dataset_args)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

        val_dataset_args = {'abnorm_feats': ab_val, 'norm_feats': nor_val, 'n_cls': n_way,\
                                'n_sprt': n_shot, 'n_qry': n_query, 'n_batch': config['val_batches']}
        val_dataset = datasets.make(config['val_dataset'], **val_dataset_args)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)



        # #### Model and optimizer ###
        #import pdb; pdb.set_trace()
        if config.get('load'):
            model_sv = torch.load(config['load'])
            # model_args = {'rtfm_args': config['rtfm_args'], 'n_way': n_way, 'n_shot': n_shot, 'n_query': n_query}
            # model = models.make(config['model'], **model_args)
            model = model.load_state_dict(model_sv)
        
        else:
            model_args = {'rtfm_args': config['rtfm_args'], 'n_way': n_way, 'n_shot': n_shot, 'n_query': n_query}
            model = models.make(config['model'], **model_args)
            
        #import pdb; pdb.set_trace()    
        for name, param in model.named_parameters():
            if name.startswith('rtfm') and param.requires_grad:
                param.requires_grad = False
                

        print(f'model parameters count: {count_params(model)}') # need to check this later
        
        g_optimizer, g_lr_scheduler = utils.make_optimizer(
            model.g.parameters(),
            config['g_optimizer'], **config['g_optimizer_args'])

 
        best_acc = 0
        max_epoch = config['max_epoch']
        save_epoch = config['save_epoch']
        save_path = config['save_path']
        timer_used = utils.Timer()
        timer_epoch = utils.Timer()

        result_keys = ['tl', 'ta', 'vl', 'va']
        train_log = dict()

        for k in result_keys:
            train_log[k] = []

        # import pdb; pdb.set_trace()
        for epoch in range(1, max_epoch + 1):
            timer_epoch.s()
            aves = {k: utils.Averager() for k in result_keys}
            model.train()

            for batch in train_loader:
                # import pdb; pdb.set_trace()
                predictions, labels, loss = model(batch)
                acc = utils.compute_acc(predictions, labels)

                g_optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()

                # import pdb; pdb.set_trace()
                aves['tl'].add(loss)
                aves['ta'].add(acc)

            model.eval()

            for batch in val_loader:
                with torch.no_grad():
                    predictions, labels, loss = model(batch)
                    acc = utils.compute_acc(predictions, labels)

                aves['vl'].add(loss)
                aves['va'].add(acc)

            if g_lr_scheduler is not None:
                g_lr_scheduler.step()

            # import pdb; pdb.set_trace()
            for k, v in aves.items():
                aves[k] = v.item()
                train_log[k].append(aves[k])

            # import pdb; pdb.set_trace()
            t_epoch = utils.time_str(timer_epoch.t())
            t_used = utils.time_str(timer_used.t())
            t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

            utils.log('fold {}, epoch {}, train_loss|train_accuracy: {:.4f}|{:.4f},'
                        'val_loss|val_accuracy: {:.4f}|{:.4f}, {} {}/{}'.format(fold,
                        epoch, aves['tl'], aves['ta'], aves['vl'], aves['va'],
                        t_epoch, t_used, t_estimate))

            save_obj = {'fold': fold,
                        'epoch': epoch,
                        'model': config['model'],
                        'model_args': model_args,
                        'model_sd': model.state_dict(),
                        'g_optimizer': config['g_optimizer'],
                        'g_optimizer_args': config['g_optimizer_args'],
                        'g_optimizer_sd': g_optimizer.state_dict(),
                        't_loss': aves['tl'],
                        't_accuracy': aves['ta'],
                        'v_loss': aves['vl'],
                        'v_accuracy': aves['va']}

            
            # import pdb; pdb.set_trace()
            tb.add_scalars('Loss', {'loss/train': aves['tl'], 'loss/val': aves['vl']}, epoch)
            tb.add_scalars('Accuracy', {'acc/train': aves['ta'], 'acc/val': aves['va']}, epoch)
            
            
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            torch.save(train_log, os.path.join(save_path, 'train_log.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] >= best_acc:
                torch.save(save_obj, os.path.join(save_path, 'best_va_model.pth'))
                best_acc = aves['va']

            with open(os.path.join(save_path, 'log_xavier_0.001_0.0005_5.txt'), 'a') as writer:
                writer.write('Fold: ' + str(fold) + ' Epoch: ' + str(epoch) + ' tr_loss: ' + '{0:.4f}'.format(aves['tl'].item()) + ' tr_acc: ' + '{0:.4f}'.format(aves['ta'].item()) 
                + ' v_loss: ' + '{0:.4f}'.format(aves['vl'].item()) + ' v_acc: ' + '{0:.4f}'.format(aves['va'].item()) + '\n')
    
    tb.flush()

print('end')
tb.close()

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
