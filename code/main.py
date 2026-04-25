import torch
from torch.utils.data import DataLoader
import random
import utils
import train
import dataset
import test
import vq
from parse import parse_args

args = parse_args()
data_name = args.dataset
print('DICTRec is working on', data_name)

use_cuda = True
device = torch.device("cuda:" + str(args.cuda) if use_cuda and torch.cuda.is_available() else "cpu")

if args.vq:
    vq.learning(args)

lgn_dim = 64
model_name = 'lgn'
checkpoint_name = model_name + '-' + data_name + '-' + str(lgn_dim)
user_emb, item_emb = utils.read_cf_embeddings(model_name, checkpoint_name)

train_data_raw, test_data_raw, train_codebook_df, test_codebook_df, item_num, user_num = dataset.read_data(data_name)
pop_weights = utils.get_popularity(data_name, item_emb.shape[0])

# ==================== [科学防范：绝对安全的 8:1:1 动态切分] ====================
train_data, train_codebook_data = [], []
valid_data, valid_codebook_data = [], []
test_data, test_codebook_data = [], []

print("Re-constructing dataset splits to strictly follow sequential protocol with adaptive format routing...")
for i in range(len(train_data_raw)):
    user_id = train_data_raw[i].strip().split()[0]

    tr_items = train_data_raw[i].strip().split()[1:]
    te_items = test_data_raw[i].strip().split()[1:] if i < len(test_data_raw) else []

    user_cb = train_codebook_df['user_cb_id'][i]
    tr_cbs = str(train_codebook_df['item_cb_id'][i]).strip().split()
    te_cbs = str(test_codebook_df['item_cb_id'][i]).strip().split() if i < len(test_codebook_df) else []

    if len(te_items) == 1:
        full_items = tr_items + te_items
        full_cbs = tr_cbs + te_cbs
    elif len(te_items) > 1:
        full_items = te_items
        full_cbs = te_cbs
    else:
        full_items = tr_items
        full_cbs = tr_cbs

    min_len = min(len(full_items), len(full_cbs))
    full_items, full_cbs = full_items[:min_len], full_cbs[:min_len]
    l = len(full_items)
    if l < 4: continue

    split8 = max(1, int(l * 0.8))
    split9 = max(split8 + 1, int(l * 0.9))
    split9 = min(split9, l - 1)
    split8 = min(split8, split9 - 1)

    train_data.append(f"{user_id} " + " ".join(full_items[:split8]))
    train_codebook_data.append(f"{user_cb} " + " ".join(full_cbs[:split8]))

    valid_data.append(f"{user_id} " + " ".join(full_items[:split9]))
    valid_codebook_data.append(f"{user_cb} " + " ".join(full_cbs[:split9]))

    test_data.append(f"{user_id} " + " ".join(full_items))
    test_codebook_data.append(f"{user_cb} " + " ".join(full_cbs))
# =========================================================================

if not args.no_data_augment:
    train_data, train_codebook_data = utils.data_augment(train_data, train_codebook_data, shred=2,
                                                         item_limit=args.item_limit)
else:
    train_data, train_codebook_data = utils.data_construction(train_data, train_codebook_data,
                                                              item_limit=args.item_limit)

valid_data, valid_codebook_data = utils.data_construction(valid_data, valid_codebook_data, item_limit=args.item_limit)
test_data, test_codebook_data = utils.data_construction(test_data, test_codebook_data, item_limit=args.item_limit)

train_rec_dataset = dataset.LLM4RecTrainDataset(train_data, train_codebook_data, args.no_shuffle)
train_rec_loader = DataLoader(train_rec_dataset, batch_size=args.batch, shuffle=True)

valid_rec_dataset = dataset.LLM4RecDataset(valid_data, valid_codebook_data, args.no_shuffle)
valid_rec_loader = DataLoader(valid_rec_dataset, batch_size=args.batch, shuffle=False)

test_rec_dataset = dataset.LLM4RecDataset(test_data, test_codebook_data, no_shuffle=True)
test_rec_loader = DataLoader(test_rec_dataset, batch_size=args.batch, shuffle=False)

if not args.no_train:
    train.backbone(data_name, train_rec_loader, valid_rec_loader, user_emb.to(device), item_emb.to(device), item_num,
                   args, device)

test.backbone(data_name, test_rec_loader, user_emb.to(device), item_emb.to(device), item_num, args, device, pop_weights)