from dataclasses import dataclass, field


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    balanced_accuracy: list[float] = field(default_factory=list)
    roc_auc: list[float] = field(default_factory=list)
    average_precision: list[float] = field(default_factory=list)

    def append(self, loss, accuracy, balanced_accuracy, roc_auc, average_precision):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.balanced_accuracy.append(balanced_accuracy)
        self.roc_auc.append(roc_auc)
        self.average_precision.append(average_precision)
