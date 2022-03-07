# DST_als: Algorithm integrated of *Dialogue State Tracking*

An indispensable component in task-oriented dialogue systems is
the dialogue state tracker, which keeps track of users’ intentions in
the course of conversation. The typical approach towards this goal
is to fill in multiple pre-defined slots that are essential to complete
the task. To this end, DST_cls aims to provide easy implementations with unified interfaces to facilitate the research in Dialogue State Tracking.

---

NOTE: *DST_als* is still in the early stages and the API will likely continue to change. 

If you are interested in this project, don't hesitate to contact me or make a PR directly.


# 🚀 Installation

Please make sure you have installed [PyTorch](https://pytorch.org) and [Transformers](https://huggingface.co/docs/transformers/index).

```bash
# Comming soon
pip install -U dstals
```

or

```bash
# Recommended now
git clone https://github.com/cyberfish1120/DST_als.git && cd DST_als
pip install -e . --verbose
```

where `-e` means "editable" mode so you don't have to reinstall every time you make changes.

# ⚡ Get Started

## A simple example of how to use

```python
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

```



# 👀 Implementations

In detail, the following methods are currently implemented:

## based-ontology

| Methods             | Venue                                                        |
| ------------------- | ------------------------------------------------------------ |
| **STAR**            | *Ye, Fanghua, et al.* **Slot Self-Attentive Dialogue State Tracking** [📝](https://dl.acm.org/doi/abs/10.1145/3442381.3449939) Proceedings of the Web Conference 2021. 2021. |

## open-ontology

| Methods                   | Venue                                                        |
| ------------------------- | ------------------------------------------------------------ |
| **SomDST**            | *Kim, Sungdong, et al.* **Efficient dialogue state tracking by selectively overwriting memory** [📝](https://arxiv.org/abs/1911.03906) arXiv preprint arXiv:1911.03906 (2019). |

