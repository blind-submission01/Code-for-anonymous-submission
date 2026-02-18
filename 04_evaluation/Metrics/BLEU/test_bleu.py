"""测试 BLEU 计算是否正确"""
from sacrebleu.metrics import BLEU

# 测试数据
pred = "The `itermonomials` function incorrectly filters monomials based on the **maximum exponent** of individual variables rather than the **total degree** of the monomial. When `min_degrees` is specified, the code checks if `max(powers.values()) >= min_degree`, which erroneously excludes valid monomials whose total degree meets the requirement but has no single variable with exponent ≥ `min_degrees`."

ref = "`itermonomials` is documented to treat `min_degrees` (when given as an integer) as a lower bound on the **total degree** of a monomial. However, in `sympy/polys/monomials.py` inside `itermonomials`, the filtering logic for `min_degree` checks `max(powers.values()) >= min_degree` (see the provided patch context around the two occurrences near lines ~127 and ~139). Here, `powers` is a per-variable exponent counter for the candidate monomial, so `max(powers.values())` measures the **largest single-variable exponent**, not the total degree.\n\nAs a result, mixed monomials with the correct total degree (e.g. `x1*x2**2`, which has exponents `{x1: 1, x2: 2}` and total degree 3) are incorrectly rejected when `min_degree=3`, because `max(...)` is 2. Only “pure power” monomials like `x1**3` pass, which matches the reported output `[x3**3, x2**3, x1**3]` and explains why monomials remain missing even when `max_degrees > min_degrees`."

bleu = BLEU()

# 方法1：单条计算
score1 = bleu.corpus_score([pred], [[ref]]).score
print(f"单条计算: {score1:.4f}")

# 方法2：使用 evaluate 库验证
try:
    from evaluate import load
    bleu_eval = load("sacrebleu")
    result = bleu_eval.compute(predictions=[pred], references=[[ref]])
    print(f"evaluate 库: {result['score']:.4f}")
except:
    print("evaluate 库不可用")

# 方法3：sentence_score (单句)
score3 = bleu.sentence_score(pred, [ref]).score
print(f"sentence_score: {score3:.4f}")