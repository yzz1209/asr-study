from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DtwResult:
    """
    DTW 的输出结果。

    cost: 最优对齐路径的总代价（越小越相似）
    path: 对齐路径 [(i,j), ...]，表示 seq1[i] 与 seq2[j] 在路径上对齐
    """
    cost: float
    path: list[tuple[int, int]]


def _default_dist(a: float, b: float) -> float:
    # 默认的点对点距离：绝对值差。实际使用中可以换成更复杂的距离函数。
    return abs(a - b)


def dtw_align(
    seq1: Sequence[T],
    seq2: Sequence[T],
    dist: Callable[[T, T], float] | None = None,
) -> DtwResult:
    """
    Dynamic Time Warping（DTW）：对齐两个长度可能不同的序列。

    适用场景（直觉）：
    - 两段序列描述同一个过程，但局部“快/慢”不同（允许拉伸/压缩）
    - 希望得到一条整体一致的对齐映射（path），而不是只做局部贪心匹配

    定义：
    - dp[i][j]：seq1 的前 i 个元素与 seq2 的前 j 个元素对齐的最小总代价
    - 转移允许三种前驱（对应“时间规整”能力）：
      1) (i-1, j)   ：seq1 前进，seq2 不动（seq1 的一个点对齐多个 seq2 点）
      2) (i,   j-1) ：seq2 前进，seq1 不动（seq2 的一个点对齐多个 seq1 点）
      3) (i-1, j-1) ：两边同时前进（一对一对齐）

    代价：
    - 每走到 (i,j) 需要支付 dist(seq1[i-1], seq2[j-1]) 的“局部代价”
    - 总代价为路径上局部代价的累加
    """
    if dist is None:
        dist = _default_dist  # type: ignore[assignment]

    n = len(seq1)
    m = len(seq2)
    if n == 0 or m == 0:
        # 任一序列为空时，只有两者都空才算“完全对齐”，否则代价视为无穷大。
        return DtwResult(cost=inf if (n != m) else 0.0, path=[])

    # dp 初始化为 inf，表示“当前还不可达/尚未被更优路径更新”。
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    # bt 记录最优路径的回溯指针：bt[i][j] = (prev_i, prev_j)。
    bt: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 当前格子的局部代价（seq1 的第 i-1 个元素 与 seq2 的第 j-1 个元素的距离）
            cost = float(dist(seq1[i - 1], seq2[j - 1]))
            prevs: Iterable[tuple[float, tuple[int, int]]] = (
                (dp[i - 1][j], (i - 1, j)),
                (dp[i][j - 1], (i, j - 1)),
                (dp[i - 1][j - 1], (i - 1, j - 1)),
            )
            # 选择 dp 最小的前驱，然后加上局部代价。
            best_prev_cost, best_prev = min(prevs, key=lambda x: x[0])
            dp[i][j] = best_prev_cost + cost
            bt[i][j] = best_prev

    # 从 (n,m) 沿 bt 回溯，得到对齐路径（注意 path 里用的是 0-based 的元素下标）。
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = bt[i][j]
        if prev is None:
            break
        i, j = prev
    path.reverse()
    return DtwResult(cost=dp[n][m], path=path)


def _main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq1", required=True, help="Comma-separated numbers, e.g. 1,2,3")
    parser.add_argument("--seq2", required=True, help="Comma-separated numbers, e.g. 2,2,4")
    args = parser.parse_args()

    s1 = [float(x) for x in args.seq1.split(",") if x.strip() != ""]
    s2 = [float(x) for x in args.seq2.split(",") if x.strip() != ""]
    res = dtw_align(s1, s2)
    sys.stdout.write(json.dumps({"cost": res.cost, "path": res.path}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    _main()
