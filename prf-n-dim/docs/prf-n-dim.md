# PrfNdimの内部仕様

## PrfNdimの形式的定義

PrfNdim (Primitive Recursive Function of N dimension)とは`Z, S, P(n, i), C, R`(`n`, `i`は自然数)を用いて以下のように定義されるCFGのようなものであり、
原始再帰関数をもとにしている。これはStrictPrfに対してRのステップ関数が複数あることを許した体系である。


```
1. (零関数) Expr ::= Z()
2. (後者関数) Expr ::= S()
3. (射影作用素) Expr ::= P(n, i) (n, iはn>=iなる任意の自然数)
4. (合成作用素) Expr ::= C(Expr, Expr, ...,) (...は任意の有限個のExpr)
5. (原始再帰作用素) Expr ::= R(Expr, Expr, Expr ...) (...は任意の有限個のExpr)
```
PrfNdimの例を挙げる

- `C(S(), S())`
- `C(S(), C(S(), Z()))`
- `R(P(1, 1), C(S, P(3, 3)))`

ただし、語ごとにarityという値があり、全ての語はarityに関して以下の全てを満たさなければならない。

```
1. Z()のarityはNoneである
2. S()のarityは1である
3. P(n, i)のarityはnである
4. C(Expr1, Expr2, ...,Expri)において、Expr1のarityはiに等しくなければならず、Expr2, Expr3...,Expriのそれぞれのarityは全て等しくなければならない(この値をdとする)。このとき、C(Expr1, Expr2, ...,Expri)のarityはdになる。
5. R(Expr_t, Expr_a1, Expr_a2, Expr_an, Expr_b1, Expr_b2, ...Expr_bn)において引数のarityは以下を満たさなければならない。このときRのarityはnに
- Expr_bi.arity のarityをdとすると、Expr_ai.arityはn + d + 1 であり、Expr_t.arityはnである
```

ただし、arity=Noneとは1, 2, ...の全ての自然数の中で適当な数をarityに取れるワイルドカードのような役割である。


# PrfNdimの意味的定義

上で述べた言語PrfNdimの語は自然数関数$\mathbb{N}^n \to \mathbb{N}$ と対応しており、各語のarityはその自然数関数の引数の数を表している。具体的に各規則は以下の意味を持つ
- `Z()`: 引数を任意個取り、常にゼロを返す関数
- `S()`: 1つの引数を取り、引数+1を返す関数
- `P(n, i) (n, iはn>=iなる自然数)`: 引数をn個数とり、i番目の引数の値を返す関数
- `C(Expr1, Expr2, ..., Expri)`: 第2引数以降の関数の出力を第一引数が表す関数の入力に1つずつ入れた関数(下に詳述)。
- `R(Expr_t, Expr_a1, ...,Expr_an, Expr_b1, ...,Expr_bn)`: 引数を2n+1個とり、原始再帰法における底をExpr_biが、繰り返し部分をExpr_aiが担う。このままでは引数がn個出てしまうので、Expr_tに全てを代入することで引数を1つにする。
