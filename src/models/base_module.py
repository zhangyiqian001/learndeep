from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.modules.loss import *
from torch.optim import *
from timm.optim import Adan
from torchmetrics import *

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

METRIC = {
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


def define_loss(config):
    return LOSS[config.LOSS.TYPE]()


def define_metric(config):
    args = OmegaConf.to_container(config.METRIC.ARGS)
    args = {key.lower(): value for key, value in args.items()}
    args['num_classes'] = config.NUM_CLASSES
    return METRIC[config.METRIC.TYPE](**args)


class BaseModelModule(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(args[0])

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs)
        if "classification" in self.config.TASK:
            output = torch.nn.functional.softmax(output, dim=1)
        loss = define_loss(self.config.MODEL)(output, target)
        metric = define_metric(self.config.MODEL).to(self.device)
        acc = metric(output.argmax(1), target.argmax(1))
        values = {"loss": loss, "acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs)
        if "classification" in self.config.TASK:
            output = torch.nn.functional.softmax(output, dim=1)
        loss = define_loss(self.config.MODEL)(output, target)
        metric = define_metric(self.config.MODEL).to(self.device)
        acc = metric(output.argmax(1), target.argmax(1))
        values = {"loss": loss, "acc": acc}
        self.log_dict(values, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch
        output = self(inputs)
        if "classification" in self.config.TASK:
            output = torch.nn.functional.softmax(output, dim=1)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        args = OmegaConf.to_container(self.config.MODEL.OPTIM.ARGS)
        args = {key.lower(): value for key, value in args.items()}
        args['params'] = self.model.parameters()
        return OPTIM[self.config.MODEL.OPTIM.TYPE](**args)
