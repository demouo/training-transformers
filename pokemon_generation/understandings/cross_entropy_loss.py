import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# criterion = CrossEntropyLoss
criterion = F.cross_entropy
# criterion(logits, labels)
# shape: [N, C], [N]

# ---- Understanding ----
logits = torch.tensor(
    [[2, 1, 0.1, -1.0], [0.5, 2.5, 0.3, 0.2], [1.2, 0.7, 2.1, 0.1]]
)  # shape: [3, 4]

"""
Lables mean:
    - 1: the first one is idx=1 (1 choosen)
    - 3: the second one is idx=3 (0.3 choosen)
    - 2: the third one is idx=2 (2.1 choosen)
"""
labels = torch.tensor([1, 3, 2])  # shape: [3]
print(criterion(logits, labels))  # tensor(1.5427)

# ---- Let us calculate by hand ----
# logits ---> .softmax ---> .log ---> outputs
s_logit = F.log_softmax(logits, dim=1)
print(-s_logit)
"""
tensor([[0.4493, 1.4493, 2.3493, 3.4493],
        [2.2974, 0.2974, 2.4974, 2.5974],
        [1.4814, 1.9814, 0.5814, 2.5814]])

    - 1: the first one is idx=1 (1.4493 choosen)
    - 3: the second one is idx=3 (2.5974 choosen)
    - 2: the third one is idx=2 (0.5814 choosen)
"""
print((1.4493 + 2.5974 + 0.5814) / 3)  # 1.5427
