from typing import Any

import torch
from lightning.pytorch import LightningModule
from timm.optim import Adan
from torch import Tensor
from torch.nn.modules.loss import *
from torch.optim import *
from torchmetrics import *

OPTIM = {
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "Adan": Adan,
    "AdamW": AdamW,
    "SparseAdam": SparseAdam,
    "Adamax": Adamax,
    "ASGD": ASGD,
    "SGD": SGD,
    "RAdam": RAdam,
    "Rprop": Rprop,
    "RMSprop": RMSprop,
    "NAdam": NAdam,
    "LBFGS": LBFGS,
}

LOSS = {
    'L1Loss': L1Loss,
    'NLLLoss': NLLLoss,
    'NLLLoss2d': NLLLoss2d,
    'PoissonNLLLoss': PoissonNLLLoss,
    'GaussianNLLLoss': GaussianNLLLoss,
    'KLDivLoss': KLDivLoss,
    'MSELoss': MSELoss,
    'BCELoss': BCELoss,
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'HingeEmbeddingLoss': HingeEmbeddingLoss,
    'MultiLabelMarginLoss': MultiLabelMarginLoss,
    'SmoothL1Loss': SmoothL1Loss,
    'HuberLoss': HuberLoss,
    'SoftMarginLoss': SoftMarginLoss,
    'CrossEntropyLoss': CrossEntropyLoss,
    'MultiLabelSoftMarginLoss': MultiLabelSoftMarginLoss,
    'CosineEmbeddingLoss': CosineEmbeddingLoss,
    'MarginRankingLoss': MarginRankingLoss,
    'MultiMarginLoss': MultiMarginLoss,
    'TripletMarginLoss': TripletMarginLoss,
    'TripletMarginWithDistanceLoss': TripletMarginWithDistanceLoss,
    'CTCLoss': CTCLoss
}

METRICS = {
    "Accuracy": Accuracy,
    "AUROC": AUROC,
    "AveragePrecision": AveragePrecision,
    "BLEUScore": BLEUScore,
    "BootStrapper": BootStrapper,
    "CalibrationError": CalibrationError,
    "CatMetric": CatMetric,
    "ClasswiseWrapper": ClasswiseWrapper,
    "CharErrorRate": CharErrorRate,
    "CHRFScore": CHRFScore,
    "ConcordanceCorrCoef": ConcordanceCorrCoef,
    "CohenKappa": CohenKappa,
    "ConfusionMatrix": ConfusionMatrix,
    "CosineSimilarity": CosineSimilarity,
    "CramersV": CramersV,
    "CriticalSuccessIndex": CriticalSuccessIndex,
    "Dice": Dice,
    "TweedieDevianceScore": TweedieDevianceScore,
    "ErrorRelativeGlobalDimensionlessSynthesis": ErrorRelativeGlobalDimensionlessSynthesis,
    "ExactMatch": ExactMatch,
    "ExplainedVariance": ExplainedVariance,
    "ExtendedEditDistance": ExtendedEditDistance,
    "F1Score": F1Score,
    "FBetaScore": FBetaScore,
    "FleissKappa": FleissKappa,
    "HammingDistance": HammingDistance,
    "HingeLoss": HingeLoss,
    "JaccardIndex": JaccardIndex,
    "KLDivergence": KLDivergence,
    "MeanAbsoluteError": MeanAbsoluteError,
    "MeanAbsolutePercentageError": MeanAbsolutePercentageError,
    "MeanMetric": MeanMetric,
    "MeanSquaredError": MeanSquaredError,
    "MeanSquaredLogError": MeanSquaredLogError,
    "Metric": Metric,
    "Precision": Precision,
    "R2Score": R2Score,
    "Recall": Recall,
    "RetrievalFallOut": RetrievalFallOut,
    "RetrievalHitRate": RetrievalHitRate,
    "RetrievalMAP": RetrievalMAP,
    "RetrievalMRR": RetrievalMRR,
    "RetrievalNormalizedDCG": RetrievalNormalizedDCG,
    "RetrievalPrecision": RetrievalPrecision,
    "RetrievalRecall": RetrievalRecall,
    "RetrievalRPrecision": RetrievalRPrecision,
    "RetrievalPrecisionRecallCurve": RetrievalPrecisionRecallCurve,
    "RetrievalRecallAtFixedPrecision": RetrievalRecallAtFixedPrecision,
    "ROC": ROC,
    "RootMeanSquaredErrorUsingSlidingWindow": RootMeanSquaredErrorUsingSlidingWindow,
    "RunningMean": RunningMean,
    "RunningSum": RunningSum,
    "SacreBLEUScore": SacreBLEUScore,
    "SignalDistortionRatio": SignalDistortionRatio,
    "ScaleInvariantSignalDistortionRatio": ScaleInvariantSignalDistortionRatio,
    "ScaleInvariantSignalNoiseRatio": ScaleInvariantSignalNoiseRatio,
    "SignalNoiseRatio": SignalNoiseRatio,
    "SQuAD": SQuAD,
    "StatScores": StatScores,
    "SumMetric": SumMetric,
    "SymmetricMeanAbsolutePercentageError": SymmetricMeanAbsolutePercentageError,
    "TheilsU": TheilsU,
    "TotalVariation": TotalVariation,
    "TranslationEditRate": TranslationEditRate,
    "TschuprowsT": TschuprowsT,
    "UniversalImageQualityIndex": UniversalImageQualityIndex,
    "WeightedMeanAbsolutePercentageError": WeightedMeanAbsolutePercentageError,
    "WordErrorRate": WordErrorRate,
    "WordInfoLost": WordInfoLost,
    "WordInfoPreserved": WordInfoPreserved,
}

class BaseModelModule(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.loss = LOSS[self.config['loss']](**self.config['loss_args'])
        self.metrics = METRICS[self.config['metrics']](**self.config['metrics_args']).to(self.device)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(args[0])

    def training_step(self, batch, batch_idx: int) -> Tensor:
        inputs, target = batch['inputs'], batch['targets']
        output = self(inputs)
        loss = self.loss(output, target)
        acc = self.metrics(output.argmax(1), target)
        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        inputs, target = batch['inputs'], batch['targets']
        output = self(inputs)
        loss = self.loss(output, target)
        acc = self.metrics(output.argmax(1), target)
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx):
        inputs = batch['inputs']
        output = self(inputs)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return OPTIM[self.config['optim']](params=self.model.parameters(), **self.config['optim_args'])

    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> Any:
        result = {}
        for key,value in batch.items():
            if isinstance(value, dict):
                result[key] = {k: v.to(device) for k,v in value}
            else:
                result[key] = value.to(device)
        return result

    @classmethod
    def from_config(cls, config):
        return cls(config)