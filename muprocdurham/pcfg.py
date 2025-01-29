from triangularmap import DictTMap, TMap
from collections import defaultdict


def cat_pretty(trees, off=False, *args, **kwargs):
    if off:
        return '\n'.join([tree.pretty(*args, **kwargs) for tree in trees])
    else:
        return '\n'.join(
            list('|'.join(x) for x in zip(*[tree.pretty(*args, **kwargs).split('\n') for tree in trees]))
        )


class PrettySet(set):
    def __init__(self, *args, empty="∅", compact=True, left="{", right="}", **kwargs):
        super().__init__(*args, **kwargs)
        self.empty = empty
        self.compact = compact
        self.left = left
        self.right = right

    def __str__(self):
        if self:
            if self.compact:
                return self.left + ",".join([str(x) for x in self]) + self.right
            else:
                return str(set(self))
        else:
            return self.empty

    def __repr__(self):
        return self.__str__()


class PrettyDict(defaultdict):
    def __init__(self, *args, empty="{}", compact=True, left="{", right="}", **kwargs):
        super().__init__(*args, **kwargs)
        self.empty = empty
        self.compact = compact
        self.left = left
        self.right = right

    def __str__(self):
        if self:
            if self.compact:
                return self.left + ",".join([f"{k}:{v}" for k, v in self.items()]) + self.right
            else:
                return str(dict(self))
        else:
            return self.empty


class SetChart(TMap):
    @classmethod
    def from_dict_set_chart(cls, chart, **kwargs):
        new_chart = SetChart(chart.n, **kwargs)
        for k, v in chart.arr.items():
            for vk in v.keys():
                new_chart.arr[k].add(vk)
        return new_chart

    def __new__(cls, n, **kwargs):
        return DictTMap(n, lambda: PrettySet(**kwargs))


class DictSetChart(TMap):
    def __new__(cls, n, **kwargs):
        return DictTMap(n, lambda: PrettyDict(PrettySet))