import torch
from DST_als.data import MultiwozDataset
from DST_als.utils import set_seed
from DST_als.models import SomDST
from DST_als.training import Trainer

DATASET = 'multiwoz2.1'

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = MultiwozDataset(DATASET)

ontology = dataset.download('ontology')
train_data = dataset.download('train')
dev_data = dataset.download('dev')
test_data = dataset.download('test')


model = SomDST()
trainer = Trainer(model, device=device)
checkpoint = ModelCheckpoint()
trainer.fit(train_data, dev_data, callbacks=[checkpoint])
trainer.evaluate(test_data)








