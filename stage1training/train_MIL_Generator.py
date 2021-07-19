import sys
sys.path.append('..')
from utils.utils import *
from tqdm import tqdm
from utils.eval_utils import eval
import argparse
from stage1training.MIL_Generator import Simple_Regressor
from torch.utils.data.dataloader import DataLoader
from stage1training.SHT_MIL_dataset import SH_Train_Origin_Dataset,shanghaitech_test
import torch.nn.functional as F
import h5py

feature_type={
    'I3D_RGB':[16,1024],
    'C3D_RGB': [16,4096],
}

def topk_rank_loss(args,y_pred):
    topk_pred=torch.mean(torch.topk(y_pred.view([args.batch_size*2,args.part_num*args.part_len]),args.topk,dim=-1)[0],dim=-1,keepdim=False)

    nor_max=topk_pred[:args.batch_size]
    abn_max=topk_pred[args.batch_size:]

    err=0
    for i in range(args.batch_size):
        err+=torch.sum(F.relu(1-abn_max+nor_max[i]))
    err=err/(args.batch_size)**2

    abn_pred=y_pred[args.batch_size:]
    spar_l1=torch.mean(abn_pred)
    smooth_l2=torch.mean((abn_pred[:,:-1]-abn_pred[:,1:])**2)

    loss=err+args.lambda_1*spar_l1+args.lambda_2*smooth_l2

    return loss,err,spar_l1,smooth_l2,smooth_l2

def train(args):

    def worker_init(worked_id):
        np.random.seed(args.seed + worked_id)
        random.seed(args.seed + worked_id)

    dataset=SH_Train_Origin_Dataset(args.part_num,args.part_len,args.feature_rgb_path,args.training_txt,args.sample,None#args.Pseudo_Labels_dir+'pseudo_labels.npy'
                                    ,args.norm)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,num_workers=4,worker_init_fn=worker_init,drop_last=True)
    model=Simple_Regressor(args.size,args.dropout_rate).cuda().train()
    optimizer=torch.optim.Adagrad(model.parameters(),lr=args.lr,weight_decay=0.001)
    test_feats,test_labels,test_annos=shanghaitech_test(args.testing_txt,args.test_mask_dir,args.feature_rgb_path,args.norm)
    best_AUC = 0
    best_iter = 0
    count = 0
    for epoch in range(args.epochs):
        for norm_feats,norm_labs,abnorm_feats,abnorm_labs in dataloader:
            feats=torch.cat([norm_feats,abnorm_feats],dim=0).cuda().float().view([args.batch_size*2,args.part_num*args.part_len,args.size])
            labs=torch.cat([norm_labs,abnorm_labs],dim=0).cuda().float().view([args.batch_size*2,args.part_num*args.part_len,1])
            outputs=model(feats)
            outputs = outputs.view([args.batch_size * 2, args.part_num * args.part_len, -1])
            outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)

            loss,err,l1,_,l3=topk_rank_loss(args,outputs_mean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # if count % 10 == 0:
            print('[{}/{}]: loss {:.4f}, err {:.4f}, l1 {:.4f}， l2 {:.4f} kl {:.4f}'.format(
                count, epoch, loss, err, l1, l3,kl))
            count += 1
        dataloader.dataset.shuffle_keys()
        if epoch % 10 == 0:
            total_scores = []
            total_labels = []

            with torch.no_grad():
                model = model.eval()
                for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
                    test_feat=np.array(test_feat).reshape([-1,args.size])
                    temp_score = []
                    for i in range(test_feat.shape[0]):
                        feat = test_feat[i]
                        feat = torch.from_numpy(np.array(feat)).float().cuda().view([-1,args.size])
                        logits = model(feat)
                        score = torch.mean(logits).item()
                        temp_score.extend([score]*args.segment_len)

                    total_labels.extend(test_anno[:len(temp_score)].tolist())
                    total_scores.extend(temp_score)
            auc = eval(total_scores, total_labels,None)
            if auc > best_AUC:
                best_iter = epoch
                best_AUC = auc

            if auc>0.90:
                torch.save(model.state_dict(), args.model_path_pre + '{}_norm_{}_part_num_{}_part_len_{}_epoch_{}_AUC_{}.pth'.format(args.type,args.norm,args.part_num,args.part_len,epoch, auc))
            print('best_AUC {} at epoch {}'.format(best_AUC, best_iter))
            print('===================')
            model = model.train()

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)

    parser.add_argument('--topk',type=int,default=7)
    # part num should consider the average len of the video
    parser.add_argument('--part_num', type=int, default=32)
    parser.add_argument('--part_len', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=3001)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--dropout_rate', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')

    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--clip', type=float, default=4.0)

    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)

    parser.add_argument('--machine',type=str,default='data0')

    parser.add_argument('--feature_rgb_path', type=str, default='/jiachang/SHT_I3D.h5')
    parser.add_argument('--model_path_pre', type=str, default='/jiachang/Weakly_Supervised_VAD/models/shanghaitech_single')
    parser.add_argument('--training_txt',type=str,default='/jiachang/Weakly_Supervised_VAD/Datasets/SH_Train_new.txt')
    parser.add_argument('--testing_txt',type=str,default='/jiachang/Weakly_Supervised_VAD/Datasets/SH_Test_NEW.txt')
    parser.add_argument('--test_mask_dir',type=str,default='/jiachang/Weakly_Supervised_VAD/Datasets/test_frame_mask/')
    parser.add_argument('--model_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Datasets/MIL.pth')
    parser.add_argument('--Pseudo_Labels_dir',type=str,default='/jiachang/Pseudo_Labels_MultiViews/SHT_PLs_I3D_iter_1')

    parser.add_argument('--generate_PL',action='store_true',dest='PL')
    parser.set_defaults(PL=False)

    parser.add_argument('--smooth_len',type=int,default=5)

    args = parser.parse_args()
    args.machine='/'+args.machine
    args.feature_rgb_path=args.machine+args.feature_rgb_path
    args.model_path_pre=args.machine+args.model_path_pre
    args.training_txt=args.machine+args.training_txt
    args.testing_txt=args.machine+args.testing_txt
    args.test_mask_dir=args.machine+args.test_mask_dir
    args.model_path=args.machine+args.model_path
    args.Pseudo_Labels_dir=args.machine+args.Pseudo_Labels_dir

    return args

def Augment_Pseudo_Label_Generate(args):
    model = Simple_Regressor(feature_type[args.type][1]).cuda().eval()
    test_keys=[]
    for line in open(args.testing_txt).readlines():
        test_keys.append(line.strip().split(',')[0].split('.')[0]+'.npy')

    model.load_state_dict(torch.load(args.model_path))
    scores_dict = {}
    test_scores_dict = {}

    with h5py.File(args.feature_rgb_path, 'r') as h5:
        keys=list(h5.keys())
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            for key in keys:
                feat = h5[key][:]
                feat=np.array(feat)
                if args.norm==2:
                    feat=feat/np.linalg.norm(feat,axis=-1,keepdims=True)
                feat = torch.from_numpy(feat).cuda().float()
                if feat.shape[0]>1000:
                    scores=[]
                    length=feat.shape[0]//1000+1
                    for i in range(length):
                        if i==length-1:
                            f=feat[i*1000:]
                        else:
                            f=feat[i*1000:(i+1)*1000]
                        scores.append(model(f).detach().cpu().numpy())
                    scores=np.vstack(scores)
                else:
                    scores = model(feat).detach().cpu().numpy()
                scores=scores.squeeze()
                if key in test_keys:
                    test_scores_dict[key]=scores
                else:
                    scores_dict[key]=scores
    np.save(args.Pseudo_Labels_dir+'/train_results.npy', scores_dict)
    np.save(args.Pseudo_Labels_dir+'/test_results.npy', test_scores_dict)

    Augment_Pseudo_Label_Refilement(args)

def smooth(y,box_size):
    assert box_size%2==1, 'The bosx size should be ood'
    box=np.ones(box_size)/box_size
    y=np.array([y[0]]*(box_size//2)+y.tolist()+[y[-1]]*(box_size//2))
    y_smooth=np.convolve(y,box,mode='valid')
    return y_smooth

def Augment_Pseudo_Label_Refilement(args):
    lines=open(args.training_txt,'r').readlines()
    train_score_dict=np.load(args.Pseudo_Labels_dir+'/train_results.npy',allow_pickle=True).tolist()
    output_labels_dict={}
    for line in tqdm(lines):
        line_split = line.strip().split(',')
        label = int(line_split[-1])
        key = line_split[0] + '.npy'
        score=train_score_dict[key]
        if label==1:
            score=smooth(score,args.smooth_len)
            score=smooth(score,args.smooth_len)
            if score.shape[0]>=2:
                score=score-score.min()
                score=score/score.max()
            output_labels_dict[key]=score
        else:
            score=smooth(score,args.smooth_len)
            score=smooth(score,args.smooth_len)
            output_labels_dict[key]=score
    np.save(args.Pseudo_Labels_dir+'/pseudo_labels.npy',output_labels_dict)
    print('finished!')

if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_seeds(args.seed)
    # first step to train the MIL generator
    if not args.PL:
        train(args)
    # then generate pseudo labels after the MIL generator trained, use the command below
    else:
        Augment_Pseudo_Label_Generate(args)
