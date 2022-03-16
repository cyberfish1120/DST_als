# 💡 DST_als: *Dialog State Tracking* algorithm integration

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

## A simple example

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


## based-ontology

| Methods             | Venue                                                        | Joint Acc on Mwz2.1 | Implemented |
| ------------------- | ------------------------------------------------------------ | :-----------------: | :---------: |
| **STAR**            | *Ye, Fanghua, et al.* [**Slot Self-Attentive Dialogue State Tracking** 📝](https://dl.acm.org/doi/abs/10.1145/3442381.3449939) Proceedings of the Web Conference 2021. 2021. | 56.36% | ✔ |
| **HJST**            | *Eric, Mihail, et al.* [**Multiwoz 2.1: Multi-domain dialogue state corrections and state tracking baselines**📝](https://arxiv.org/abs/1907.01669) (2019). | 35.55% | ✔ |
| **FJST**            | *Eric, Mihail, et al.* [**Multiwoz 2.1: Multi-domain dialogue state corrections and state tracking baselines**📝](https://arxiv.org/abs/1907.01669) (2019). | 38.00% | ✔ |
| **SUMBT**           | *Lee, Hwaran, Jinsik Lee, and Tae-Yoon Kim.* [**Sumbt: Slot-utterance matching for universal and scalable belief tracking**📝](https://arxiv.org/abs/1907.07421) arXiv preprint arXiv:1907.07421 (2019). | **-** |  |
| **HyST**            | *Goel, Rahul, Shachi Paul, and Dilek Hakkani-Tür.* [**Hyst: A hybrid approach for flexible and accurate dialogue state tracking**📝](https://arxiv.org/abs/1907.00883) arXiv preprint arXiv:1907.00883 (2019). | 38.10% |  |
| **DS-DST**          | *Zhang, Jian-Guo, et al.* [**Find or classify? dual strategy for slot-value predictions on multi-domain dialog state tracking**📝](https://arxiv.org/abs/1910.03544) arXiv preprint arXiv:1910.03544 (2019). | 51.21% |  |
| **DSTQA**            | *Zhou, Li, and Kevin Small.* [**Multi-domain dialogue state tracking as dynamic knowledge graph enhanced question answering**📝](https://arxiv.org/abs/1911.06192) arXiv preprint arXiv:1911.06192 (2019).| 51.17% | ✔ |

## open-ontology

| Methods                   | Venue                                                        | Joint Acc on Mwz2.1 | Implemented |
| ------------------------- | ------------------------------------------------------------ | :-----------------: | :---------: |
| **SomDST**            | *Kim, Sungdong, et al.* [**Efficient dialogue state tracking by selectively overwriting memory** 📝](https://arxiv.org/abs/1911.03906) arXiv preprint arXiv:1911.03906 (2019). | 53.01% | ✔ |
| **ReInf**             | *Liao, Lizi, et al* [**Multi-domain Dialogue State Tracking with Recursive Inference**📝](https://dl.acm.org/doi/abs/10.1145/3442381.3450134) Proceedings of the Web Conference 2021. 2021.| 58.3% |  |
| **DST-Reader**            | *Gao, Xiang, et al.* [**Jointly optimizing diversity and relevance in neural response generation**📝](https://arxiv.org/abs/1902.11205) arXiv preprint arXiv:1902.11205 (2019). | 36.40% |  |
| **TRADE**            | *Wu, Chien-Sheng, et al.* [**Transferable multi-domain state generator for task-oriented dialogue systems**📝](https://arxiv.org/abs/1905.08743) arXiv preprint arXiv:1905.08743 (2019). | 45.60% | ✔ |
| **COMER**            | *Ren, Liliang, Jianmo Ni, and Julian McAuley.* [**Scalable and accurate dialogue state tracking via hierarchical sequence generation**📝](https://arxiv.org/abs/1909.00754) arXiv preprint arXiv:1909.00754 (2019). | **-** | ✔ |
| **NADST**            | *Le, Hung, Richard Socher, and Steven CH Hoi.* [**Non-autoregressive dialog state tracking**📝](https://arxiv.org/abs/2002.08024) arXiv preprint arXiv:2002.08024 (2020). | 49.04% | ✔ |
| **SAS**            | *Hu, Jiaying, et al.* [**SAS: Dialogue state tracking via slot attention and slot information sharing**📝](https://aclanthology.org/2020.acl-main.567/) Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020. | **-** |  |
| **CSFN-DST**            | *Zhu, Su, et al.* [**Efficient context and schema fusion networks for multi-domain dialogue state tracking**📝](https://arxiv.org/abs/2004.03386) arXiv preprint arXiv:2004.03386 (2020). | 52.88% |  |
| **Graph-DST**            | *Zeng, Yan, and Jian-Yun Nie.* [**Multi-domain dialogue state tracking based on state graph**📝](https://arxiv.org/abs/2010.11137) arXiv preprint arXiv:2010.11137 (2020). | 53.85% |  |