import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from torch.nn import Module

from modules.base_module import BaseDistillModule
from .vit_module import ViTModel


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class DistillMixin:
    def forward(self, img, distill_token = None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)

        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out


class DistillableViT(DistillMixin, ViTModel):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViTModel(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

# class DistillableT2TViT(DistillMixin, T2TViT):
#     def __init__(self, *args, **kwargs):
#         super(DistillableT2TViT, self).__init__(*args, **kwargs)
#         self.args = args
#         self.kwargs = kwargs
#         self.dim = kwargs['dim']
#         self.num_classes = kwargs['num_classes']
#
#     def to_vit(self):
#         v = T2TViT(*self.args, **self.kwargs)
#         v.load_state_dict(self.state_dict())
#         return v
#
#     def _attend(self, x):
#         x = self.dropout(x)
#         x = self.transformer(x)
#         return x
#
# class DistillableEfficientViT(DistillMixin, EfficientViT):
#     def __init__(self, *args, **kwargs):
#         super(DistillableEfficientViT, self).__init__(*args, **kwargs)
#         self.args = args
#         self.kwargs = kwargs
#         self.dim = kwargs['dim']
#         self.num_classes = kwargs['num_classes']
#
#     def to_vit(self):
#         v = EfficientViT(*self.args, **self.kwargs)
#         v.load_state_dict(self.state_dict())
#         return v
#
#     def _attend(self, x):
#         return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(Module):
    def __init__(
            self,
            *,
            teacher,
            student,
            temperature = 1.,
            alpha = 0.5,
            hard = False,
            mlp_layernorm = False
    ):
        super().__init__()
        # assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'
        assert (isinstance(student, DistillableViT)) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):

        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        print(student_logits.shape)
        print(distill_logits.shape)
        print(teacher_logits.shape)
        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
                reduction = 'batchmean')
            distill_loss *= T ** 2

        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha


class DistillWrapperModule(BaseDistillModule):
    def __init__(self, model, loss, metrics):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        return {
            "inputs": batch[0],
            "targets": batch[1],
        }

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k, v in value}
            else:
                result[key] = value.to(device)
        return result
