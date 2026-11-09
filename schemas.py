from dataclasses import dataclass, field, fields

# Order defines label ids
CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia"]


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    balanced_accuracy: list[float] = field(default_factory=list)
    roc_auc: list[float] = field(default_factory=list)
    average_precision: list[float] = field(default_factory=list)
    subject_accuracy: list[float] = field(default_factory=list)
    subject_balanced_accuracy: list[float] = field(default_factory=list)
    subject_roc_auc: list[float] = field(default_factory=list)
    subject_average_precision: list[float] = field(default_factory=list)

    def append(
        self,
        loss,
        accuracy,
        balanced_accuracy,
        roc_auc,
        average_precision,
        subject_accuracy,
        subject_balanced_accuracy,
        subject_roc_auc,
        subject_average_precision,
    ):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.balanced_accuracy.append(balanced_accuracy)
        self.roc_auc.append(roc_auc)
        self.average_precision.append(average_precision)
        self.subject_accuracy.append(subject_accuracy)
        self.subject_balanced_accuracy.append(subject_balanced_accuracy)
        self.subject_roc_auc.append(subject_roc_auc)
        self.subject_average_precision.append(subject_average_precision)


# Metric names in the order returned by utils.compute_metrics
METRICS = [f.name for f in fields(History) if f.name != "loss"]


def selection_score(history, epoch=-1):
    """Subject-level balanced accuracy, with subject-level ROC-AUC breaking ties."""
    return (history.subject_balanced_accuracy[epoch], history.subject_roc_auc[epoch])


def best_epoch(history):
    """First epoch with the highest selection score, matching the saved best model."""
    return max(
        range(len(history.loss)), key=lambda epoch: selection_score(history, epoch)
    )
