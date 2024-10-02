from torch.optim.lr_scheduler import ReduceLROnPlateau as _LRScheduler
from torchmanager.callbacks import FrequencyCallback
from torchmanager_core.protocols import Frequency, MonitorType
from torchmanager_core.typing import Any, Optional


class ReduceLROnPlateau(FrequencyCallback):
    """
    The callback to step learning rate scheduler

    * extends: `FrequencyCallback`

    - Parameters:
        - monitor: a `str` of the monitor to use for the scheduler
        - scheduler: the `torch.optim.lr_scheduler.ReduceLROnPlateau` mapping to reduce learning rate
    """
    __lr_scheduler: _LRScheduler
    monitor: str

    @property
    def scheduler(self) -> _LRScheduler:
        return self.__lr_scheduler

    def __init__(self, monitor: str, monitor_type: MonitorType, *args: Any, freq: Frequency = Frequency.EPOCH, **kwargs: Any) -> None:
        super().__init__(freq)
        self.__lr_scheduler = _LRScheduler(*args, mode=monitor_type.name.lower(), **kwargs)
        self.monitor = monitor

    def _update(self, result: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, summary: dict[str, float], val_summary: Optional[dict[str, Any]] = None) -> None:
        # update lr scheduler
        super().on_epoch_end(epoch, summary, val_summary)

    def step(self, summary: dict[str, float], val_summary: Optional[dict[str, Any]] = None) -> Any:
        # get monitor score
        monitor_score = summary[self.monitor] if val_summary is None else val_summary[self.monitor]
        self.scheduler.step(monitor_score)
