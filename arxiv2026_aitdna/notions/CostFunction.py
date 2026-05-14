from ruptures.base import BaseCost


class CostFunction(BaseCost):
    model = " "
    min_size = 2
    signal: list[int] = None
    n_tokens: int
    ideal_length: int
    n_segments: int
    length_penalty: float
    impurity_penalty: float

    def __init__(self, n_segments, length_penalty: float, impurity_penalty: float):
        self.n_segments = n_segments
        self.length_penalty = length_penalty
        self.impurity_penalty = impurity_penalty

    def fit(self, signal: list[int]):
        self.signal = signal
        self.n_tokens = len(signal)
        self.ideal_length = self.n_tokens / self.n_segments
        return self
    
    def _length_penalty(self, length):
        return max(length - self.ideal_length, self.ideal_length - length)


    def error(self, start, end):
        sub = self.signal[start:end]
        length = end - start

        n_ones = sum(sub)
        n_zeros = length - n_ones
        minority = min(n_ones, n_zeros)

        impurity = (self.impurity_penalty * minority) ** 2

        length_penalty = self.length_penalty * self._length_penalty(length)

        return impurity + length_penalty

