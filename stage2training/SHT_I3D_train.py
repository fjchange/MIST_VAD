import sys
sys.path.append('..')
from apex import amp
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.balanced_dataparallel import BalancedDataParallel
from torch.utils.data.dataloader import DataLoader
# default setting/ custom setting /options
from configs import constant,SHT_I3D_options
from datasets.dataset import *
from models.I3D_STD import *
from stage2training.losses import Weighted_BCE_Loss as WCE
from utils.eval_utils import *
from tqdm import tqdm
import random

def get_optimizer(args,model):
    params=model.parameters()
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
                                     )
    else:
        raise ValueError('must be sgd or adam')
    return optimizer

def get_model_optimizer(args,CFG):
    model=I3D_SGA_STD(args.dropout_rate,args.expand_k,
                                freeze_backbone=not (args.train_backbone), freeze_blocks=args.freeze_blocks,freeze_bn=not(args.train_bn),
                                 pretrained_backbone=args.pretrained_backbone,pretrained_path=CFG.I3D_MODEL_PATH,
                                 freeze_bn_statics=args.freeze_bn_sta).cuda()

    optimizer=get_optimizer(args,model)

    opt_level='O1'
    amp.init(allow_banned=True)
    amp.register_float_function(torch,'softmax')
    amp.register_float_function(torch,'sigmoid')
    model,optimizer=amp.initialize(model,optimizer,opt_level=opt_level,keep_batchnorm_fp32=None)
    model=BalancedDataParallel(int(args.batch_size*2*args.clip_num/len(args.gpus)*args.gpu0sz),model,dim=0,device_ids=args.gpus)

    return model,optimizer

def prepare_dataset(args):
    CFG=constant._C
    def worker_init(worked_id):
        np.random.seed(CFG.SEED+worked_id)
        random.seed(CFG.SEED+worked_id)

    Pseudo_Labels=CFG.PSEUDO_LABEL_PATH_SHT_I3D

    norm_dataset=Train_TemAug_Dataset_SHT_I3D(CFG.SHT_TRAIN_H5_PATH,CFG.SHT_TRAIN_TXT_PATH,
                                              Pseudo_Labels,args.clip_num,
                              segment_len=args.segment_len, type='Normal',hard_label=args.use_hard_label,score_segment_len=16,
                                              continuous_sampling=args.use_continuous_sampling)

    abnorm_dataset=Train_TemAug_Dataset_SHT_I3D(CFG.SHT_TRAIN_H5_PATH,CFG.SHT_TRAIN_TXT_PATH,
                                                Pseudo_Labels,args.clip_num,
                              segment_len=args.segment_len, type='Abnormal',hard_label=args.use_hard_label,score_segment_len=16,
                                                continuous_sampling=args.use_continuous_sampling)

    norm_dataloader = DataLoader(norm_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=5, worker_init_fn=worker_init,
                                 drop_last=True, )
    abnorm_dataloader = DataLoader(abnorm_dataset, batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=5, worker_init_fn=worker_init,
                                    drop_last=True, )

    test_dataset = Test_Dataset_SHT_I3D(CFG.SHT_TEST_H5_PATH, CFG.SHT_TEST_TXT_PATH,CFG.SHT_TEST_MASK_DIR,
                                        args.segment_len, ten_crop=args.ten_crop)
    test_dataloader = DataLoader(test_dataset, batch_size=42, shuffle=False, num_workers=10,
                                  worker_init_fn=worker_init, drop_last=False, )
    return norm_dataloader,abnorm_dataloader,test_dataloader

def train_SG_epoch(args,model,optimizer,criterion,norm_dataloader,abnorm_dataloader,logger,summary,iterator,epoch):
    Errs,Atten_Errs,Rmses=AverageMeter(),AverageMeter(),AverageMeter()
    for step, ((norm_frames,norm_labels),(abnorm_frames,abnorm_labels)) in enumerate(zip(norm_dataloader,abnorm_dataloader)):
        # [B,N,C,T,H,W]->[B*N,C,T,W,H]
        frames=torch.cat([norm_frames,abnorm_frames],dim=0).cuda().float()
        frames = frames.view([-1, frames.shape[2], frames.shape[3], frames.shape[4], frames.shape[5]]).cuda().float()
        # labels is with [B,N,2]->[B*N,2]
        labels=torch.cat([norm_labels,abnorm_labels],dim=0).cuda().float()
        labels = labels.view([-1, 2]).cuda().float()

        scores,feat_maps,atten_scores,_ = model(frames)

        scores=scores.view([frames.shape[0],2])[:,-1]
        atten_scores=atten_scores.view([frames.shape[0],2])[:,-1]
        labels=labels[:,-1]
        err=criterion(scores,labels)
        atten_err=criterion(atten_scores,labels)

        loss = args.lambda_base*err + args.lambda_atten*atten_err

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if iterator%args.accumulate_step==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        rmse = cal_rmse(scores.detach().cpu().numpy(), labels.unsqueeze(-1).detach().cpu().numpy())
        Rmses.update(rmse),Errs.update(err),Atten_Errs.update(atten_err)

        iterator += 1

    logger.info(
        '[{}]: err\t{:.4f}\tatten\t{:.4f}\trmse {:.4f}'.
            format( epoch, Errs,Atten_Errs,  Rmses))
    summary.add_scalar('cls_err', Errs.val(), iterator)
    summary.add_histogram('train_acc', Rmses.val(), iterator)
    Errs.reset(),Rmses.reset(),Atten_Errs.reset()

def eval_epoch(args,model,test_dataloader,logger):
    model = model.eval()
    total_labels, total_scores =  [], []

    for frames,ano_types,idxs,annos in tqdm(test_dataloader):
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()

        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]

        if args.ten_crop:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for clip, score, ano_type, idx, anno in zip(frames, scores, ano_types, idxs, annos):
            score = [score.squeeze()[1].detach().cpu().float().item()] * args.segment_len
            anno=anno.detach().numpy().astype(int)
            total_scores.extend(score)
            total_labels.extend(anno.tolist())

    return eval(total_scores,total_labels,logger)

def dir_prepare(args):
    CFG=constant._C
    get_timestamp()
    logger_dir=CFG.LOG_DIR+'{}_{}/'.format(args.MODEL,args.train)
    mkdir(logger_dir)
    param_str='SHT_{}_seed_{}_lr_{}_wd_{}_{}'.format(args.train,CFG.SEED,args.lr,args.weight_decay,
                                                                          get_timestamp())
    logger_path=logger_dir+'{}.log'.format(param_str)
    logger=get_logger(logger_path)
    logger.info('Train this model at time {}'.format(get_timestamp()))
    log_param(logger, args)
    summary_dir=CFG.SUMMARY_DIR+param_str
    mkdir(summary_dir)
    summary=SummaryWriter(summary_dir)
    model_dir=CFG.MODEL_DIR+'{}_{}/'.format(args.MODEL,args.train)
    mkdir(model_dir)
    model_path_pre=model_dir+param_str


    return logger,summary,model_path_pre

def trainer(args):
    CFG=constant._C
    logger,summary,model_path_pre=dir_prepare(args)
    norm_dataloader,abnorm_dataloader,test_dataloader=prepare_dataset(args)
    model,optimizer=get_model_optimizer(args,CFG)
    model=model.cuda().train()

    for name,p in model.module.named_parameters():
        if p.requires_grad:
            logger.info(name)

    lr_scheduler=get_lr_scheduler(args,optimizer)

    criterion=WCE(weights=args.class_reweights,label_smoothing=args.label_smoothing,eps=1e-8).cuda()

    iterator=0
    AUCs,best_epoch,best_AUC=[],0,0
    for epoch in range(args.epochs):
        if not args.train_backbone  and epoch==args.freeze_epochs:
            model.module.freeze_backbone=False
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()
        if not args.train_bn and epoch==args.freeze_bn_epochs:
            model.module.freeze_bn=False
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()

        train_SG_epoch(args,model,optimizer,criterion,norm_dataloader,abnorm_dataloader,logger,summary,iterator,epoch)

        lr_scheduler.step()
        logger.info("epoch {}, lr {}".format(epoch,optimizer.param_groups[0]['lr']))
        if epoch%10==0:
            auc=eval_epoch(args,model,test_dataloader,logger)
            AUCs.append(auc)
            summary.add_scalar('AUC', auc.item(), epoch)
            if len(AUCs) >= 5:
                if auc > best_AUC:
                    best_epoch,best_AUC =epoch,auc
                logger.info('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, auc))

            logger.info('===================')
            if auc > 0.8:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, model_path_pre + 'epoch_{}_AUC_{}.pth'.format(epoch, auc))
            model = model.train()


if __name__=='__main__':
    args=SHT_I3D_options.parse_args()
    CFG=constant._C
    set_seeds(CFG.SEED)

    show_params(args)
    trainer(args)
    show_params(args)
