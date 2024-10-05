# StrictPRFの内部仕様

## `StrictPRFの形式的定義

StrictPRF (Strict Primitive Recursive Function)とは`Z, S, P(n, i), C, R`(`n`, `i`は自然数)を用いて以下のように定義されるCFGのようなものであり、
原始再帰関数をもとにしている。

```
1. (零関数) Expr ::= Z()
2. (後者関数) Expr ::= S()
3. (射影作用素) Expr ::= P(n, i) (n, iはn>=iなる任意の自然数)
4. (合成作用素) Expr ::= C(Expr, Expr, ...,) (...は任意の有限個のExpr)
5. (原始再帰作用素) Expr ::= R(Expr, Expr)
```
StrictPRFの例を挙げる

- `C(S(), S())`
- `C(S(), C(S(), Z()))`
- `R(P(1, 1), C(S, P(3, 3)))`


ただし、語ごとにarityという値があり、全ての語はarityに関して以下の全てを満たさなければならない。

```
1. Z()のarityはNoneである
2. S()のarityは1である
3. P(n, i)のarityはnである
4. C(Expr1, Expr2, ...,Expri)において、Expr1のarityはiに等しくなければならず、Expr2, Expr3...,Expriのそれぞれのarityは等しくなければならない(この値をdとする)。このとき、C(Expr1, Expr2, ...,Expri)のarityはdになる。
5. R(Expr1, Expr2)において、Expr1のarityに2を足すとExpr2のarityにならばければいけない。またこのときR(Expr1, Expr2)のarityはExpr1のarity+1になる。
```

ただし、arityがNoneのときの対処は複雑なので、strict_prf.pyの各クラスにおけるarityメソッドを参照していただきたい。

# StrictPRFの意味的定義

上で述べた言語PRFの語は自然数関数$\mathbb{N}^n \to \mathbb{N}$ と対応しており、各語のarityはその自然数関数の引数の数を表している。具体的に各規則は以下の意味を持つ
- `Z()`: 引数を任意個取り、常にゼロを返す関数
- `S()`: 1つの引数を取り、引数+1を返す関数
- `P(n, i) (n, iはn>=iなる自然数)`: 引数をn個数とり、i番目の引数の値を返す関数
- `C(Expr1, Expr2, ..., Expri)`: 第2引数以降の関数の出力を第一引数が表す関数の入力に1つずつ入れた関数(下に詳述)。
- `R(Expr1, Expr2)`: 引数を2つとり、原始再帰法における底を第一引数、繰り返し部分を第二引数とする関数。

## C(Expr1, Expr2, ..., Expri)の詳細
Cの操作の意味的定義を詳しく述べる。PRFの語Expr1, Expr2, ..., Expri が対応している自然数関数を$f_1, f_2, ..., f_i$と表す。すると、
`C(Expr1, Expr2, ..., Expri)`は自然数関数 $f_1(f_2(x_1, x_2, ...x_d), f_3(x_1, x_2, ...x_d), ...,f_i(x_1, x_2, ...x_d))$を表している($x_1, x_2, ...,x_d$は引数)。
- これを踏まえると、Cの形式的定義のところで述べたCのarityに関するルールが妥当に見えるはずである。

## R(Expr1, Expr2) の詳細

Rは原始再帰法という計算可能性議論において重要な関数であることから、詳しい挙動は外部の情報源を参照していただきたい。
- https://ja.wikipedia.org/wiki/%E5%8E%9F%E5%A7%8B%E5%86%8D%E5%B8%B0%E9%96%A2%E6%95%B0



# 参考
- [原始再帰関数wikipe](https://ja.wikipedia.org/wiki/%E5%8E%9F%E5%A7%8B%E5%86%8D%E5%B8%B0%E9%96%A2%E6%95%B0)



