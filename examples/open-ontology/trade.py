import torch
import json
from DST_als.data import MultiwozDataset, PrepareDataset, arg_parse
from DST_als.utils import set_seed
from DST_als.models import TRADE
from DST_als.training import Trainer
from DST_als.metrics import dst_joint_acc

args = arg_parse()
print(args)

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open(args.ontology_data, 'r') as f:
    ontology = json.load(f)
train_data_raw = PrepareDataset(ontology, args.train_data_path, args).process()
train_data = MultiwozDataset(train_data_raw)
print("# train examples %d" % len(train_data_raw))

dev_data = PrepareDataset(ontology, args.dev_data_path, args).process()
print("# dev examples %d" % len(dev_data))

test_data_raw = PrepareDataset(ontology, args.test_data_path, args).process()
print("# test examples %d" % len(test_data_raw))

num_labels = [len(labels) for labels in ontology.values()]
model = TRADE(args, num_labels, args.exclude_domain)

trainer = Trainer(args, model, device=device)
trainer.fit(train_data, dev_data, metrics=dst_joint_acc)
trainer.evaluate(test_data_raw)








