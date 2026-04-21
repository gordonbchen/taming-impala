class DistLog:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._values: dict[str, float] = {}

    def avg(self, key: str, value: float) -> float:
        prev = self._values.get(key)
        if prev is None:
            self._values[key] = value
        else:
            self._values[key] = prev + self.alpha * (value - prev)
        return self._values[key]

    def pct(self, key: str, value: float) -> str:
        return f"{key} {self.avg(key, value) * 100:.1f}%"

    def scalar(self, key: str, value: float, fmt: str = ".2f", suffix: str = "") -> str:
        return f"{key} {self.avg(key, value):{fmt}}{suffix}"

    def kb(self, key: str, value: float) -> str:
        return self.scalar(key, value / 10**3, fmt=".1f", suffix=" KB")
