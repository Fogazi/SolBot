# Trailing TP Scenarios

Assumptions:
- Buy price = 1.00
- Trailing arms at +20%
- Drawdown = 10% from peak price

Formula:
- peak_price = buy * (1 + peak_gain)
- stop_price = peak_price * (1 - 0.10)
- gain_at_stop_pct = (stop_price - buy) * 100

```
Peak Gain %  Peak Price  Stop Price (-10%)  Gain at Stop %
---------    ----------  -----------------  --------------
       20        1.20              1.08               8
       30        1.30              1.17              17
       40        1.40              1.26              26
       50        1.50              1.35              35
       60        1.60              1.44              44
       70        1.70              1.53              53
       80        1.80              1.62              62
       90        1.90              1.71              71
      100        2.00              1.80              80
      110        2.10              1.89              89
      120        2.20              1.98              98
      130        2.30              2.07             107
      140        2.40              2.16             116
      150        2.50              2.25             125
      160        2.60              2.34             134
      170        2.70              2.43             143
      180        2.80              2.52             152
      190        2.90              2.61             161
      200        3.00              2.70             170
```
