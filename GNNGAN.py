import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
import models
import utils
import data_load
import random
import ipdb
# Training setting
parser = utils.get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

'''
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''
# Load data
if args.dataset == 'cora':
    adj, features, labels = data_load.load_data()
    class_sample_num = 100
    im_class_num = 3
    im_class = {1, 3, 5}
if args.dataset == 'cora_new':
    adj, features, labels = data_load.load_data_new()
    class_sample_num = 20
    im_class_num = 3
elif args.dataset == 'blog_new':
    adj, features, labels = data_load.load_data_blog_new()
    im_class_num = 14  # set it to be the number less than 100
    im_class = {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}
    class_sample_num = 20  # not used
elif args.dataset == 'BlogCatalog':
    adj, features, labels = data_load.load_data_Blog()
    im_class = {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}
    im_class_num = 14  # set it to be the number less than 100
    class_sample_num = 100  # not used
elif args.dataset == 'citeseer_new':
    adj, features, labels = data_load.load_data_citeseer_new()
    im_class_num = 1
    class_sample_num = 20
elif args.dataset == 'citeseer':
    adj, features, labels = data_load.load_data_citeseer()
    im_class_num = 1
    im_class = {0}
    class_sample_num = 100
# not used
else:
    print("no this dataset: {args.dataset}")

c_train_num = []
for i in range(labels.max().item() + 1):
        c_train_num.append(torch.sum(labels==i))

# get train, validatio, test data split
if args.dataset == 'BlogCatalog':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
if args.dataset == 'blog_new':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'cora':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'cora_new':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'citeseer':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)


# method_1: oversampling in input domain
if args.setting == 'upsampling':
    adj, features, labels, idx_train = utils.src_upsample(adj, features, labels, idx_train, portion=args.up_scale,
                                                          im_class_num=im_class_num)
if args.setting == 'smote':
    adj, features, labels, idx_train = utils.src_smote(adj, features, labels, idx_train, portion=args.up_scale,
                                                       im_class_num=im_class_num)

# Model and optimizer
# if oversampling in the embedding space is required, model need to be changed
if args.model == 'sage':
    encoder = models.Sage_En(nfeat=features.shape[1],
                             nhid=args.nhid,
                             nembed=args.nhid,
                             dropout=args.dropout)
    classifier = models.Sage_Classifier(nembed=args.nhid,
                                        nhid=args.nhid,
                                        nclass=labels.max().item() + 1,
                                        dropout=args.dropout)
elif args.model == 'gcn':
    encoder = models.GCN_En(nfeat=features.shape[1],
                            nhid=args.nhid,
                            nembed=args.nhid,
                            dropout=args.dropout)
    classifier = models.GCN_Classifier(nembed=args.nhid,
                                       nhid=args.nhid,
                                       nclass=labels.max().item() + 1,
                                       dropout=args.dropout)
elif args.model == 'GAT':
    encoder = models.GAT_En(nfeat=features.shape[1],
                            nhid=args.nhid,
                            nembed=args.nhid,
                            dropout=args.dropout)
    classifier = models.GAT_Classifier(nembed=args.nhid,
                                       nhid=args.nhid,
                                       nclass=labels.max().item() + 1,
                                       dropout=args.dropout)

decoder = models.Decoder(nembed=args.nhid,
                         dropout=args.dropout)

optimizer_en = optim.Adam(encoder.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(features, adj)
    labels_new = labels
    idx_train_new = idx_train
    adj_new = adj

    # ipdb.set_trace()
    output = classifier(embed, adj_new)
    ori_num = labels.shape[0]
    generated_G = decoder(embed)
    loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())
    loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])

    if args.setting == 'recon':##pre-train the decoder
        loss=0*loss_train+loss_rec
    else:
        loss = loss_train+0.000001*loss_rec ##finetune the decoder
    loss.backward()
    optimizer_en.step()
    optimizer_cls.step()
    if args.setting == 'recon':
        optimizer_de.step()

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:05d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    with open('F:\XQZhou\pythonProject\GNN\GraphSmote-origin/GNN-GAN/training_{}.txt'.format(args.dataset),
              'a') as file:
        file.write('Epoch: {:05d},loss_train: {:.4f},acc_train: {:.4f}, loss_val: {:.4f},acc_val: {:.4f},time: {:.4f}s\n'. \
                format(epoch + 1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(),time.time() - t))
#%%
def test(features,adj,labels):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(features, adj)
    output = classifier(embed, adj)
    test_prediction = output[idx_test]
    test_real = labels[idx_test]
    # print(np.unique(test_real))
    loss_test = F.cross_entropy(test_prediction, test_real)
    acc_test = utils.accuracy(test_prediction, test_real)

    y_pred = np.argmax(test_prediction.detach().cpu().numpy(), axis=1)
    y_real = test_real.detach().cpu().numpy()
    mcc = matthews_corrcoef(y_real, y_pred)
    f1_macro = f1_score(y_real, y_pred, average='macro')
    utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:, 2], pre='test')
    auc_score = roc_auc_score(test_real.detach().cpu(), F.softmax(test_prediction, dim=-1).detach().cpu(), multi_class="ovr", average="macro")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "mcc= {:.4f}".format(mcc),
          "f1_macro= {:.4f}".format(f1_macro),
          "auc_score= {:.4f}".format(auc_score))
    return mcc, f1_macro, auc_score

def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()
    if not os.path.exists('checkpoint/{}'.format(args.dataset)):
        os.makedirs('checkpoint/{}'.format(args.dataset))

    torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.setting,epoch, args.opt_new_G, args.im_ratio))

    return
def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset, filename),
                                map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: " + filename)

    return

#%%
# Train model
if args.load:
    filename="GNN_best"
    load_model(filename)
    mcc, f1_macro, auc_score = test(features, adj, labels)

t_total = time.time()
if args.setting == 'GAN':
    if args.finetune:
        os.environ["PYTOCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        gat_gan=torch.load('checkpoint/{}/{}_update.pth'.format(args.dataset, 'GAN'))
        features_new, labels_new, adj_new = gat_gan.fit(encoder, decoder, classifier, optimizer_en, optimizer_cls,optimizer_de,
                                                        features, labels, adj, idx_train,idx_val, args, epochs=2001,
                                                        im_class=im_class, gan_epoch=1)
        mcc, f1_macro, auc_score = test(features, adj, labels)
    else:
        os.environ["PYTOCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        gat_gan = models.GAT_GAN(batch_size=300)
        features_new, labels_new, adj_new = gat_gan.fit(encoder, decoder, classifier, optimizer_en, optimizer_cls,optimizer_de,
                                                        features, labels, adj, idx_train,idx_val, args, epochs=3001,
                                                        im_class=im_class, gan_epoch=1001)
        mcc, f1_macro, auc_score = test(features, adj, labels)
        if not os.path.exists('checkpoint/{}'.format(args.dataset)):
            os.makedirs('checkpoint/{}'.format(args.dataset))
        torch.save(gat_gan,
                   'checkpoint/{}/{}_update.pth'.format(args.dataset, 'GAN'))



if args.setting != 'GAN':
    for epoch in range(args.epochs):
        train(epoch)
        if epoch % 100 == 0:
            test(features,adj,labels)

        if epoch % 500 == 0:
            save_model(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))




