# Questions

My open questions, and the answers once found. Add any question I ask that is
worth remembering. Format: question, short answer, where to read more.

---

## Answered

**Q: What is the difference between recall@10 and nDCG@10, and why report both?**
Recall@10 compares the compressed search to the exact float32 search on the same
embeddings: "did I get the same 10 neighbours?" nDCG@10 compares to human relevance
labels (qrels): "did I get documents a person marked relevant, and are they near the
top?" A compression method could lose recall but keep nDCG (the lost neighbours were
not relevant anyway), or the other way round. Reporting both stops us fooling
ourselves. → `design.md` §2.

**Q: Why does every quantizer need a `decode()` method?**
So there is one slow-but-obviously-correct search that works for any method: rebuild
approximate vectors, then do normal float32 search. Fast paths (bit tricks) are
optional and must match the slow path in a test. Without this, every new method
would need its own search code, and bugs there would look like "results".
→ `design.md` §6.1.

**Q: What is asymmetric search?**
The query stays float32, only the stored corpus is compressed. "Symmetric" compresses
both. Asymmetric gives better quality; symmetric allows the popcount speed trick.
We always label which one a number is. → `design.md` §6.1.

**Q: How is "the evaluator is frozen" actually enforced?**
A file lists the SHA-256 hash of every file in `src/trivec/eval/`. A test recomputes
them and fails CI if they differ. The Lab runner also checks before and after each
agent experiment. Only an explicit "unfreeze eval" from me updates the hash file.
→ `design.md` §6.4.

**Q: Why Ollama as the default LLM backend?**
Free and local, so anyone can run the Lab with no API key. Paid backends are an
opt-in via config and an env var. → `design.md` §7.4.

**Q: Why not write the fast search in Rust or C from the start?**
Because I am learning and need to read every line, and because correct-then-fast is
the safe order. NumPy's bitwise ops and `np.bitwise_count` are fast enough to get
real numbers. A native extension is a later task if the numbers justify it.
→ `design.md` §8.

**Q: Recall@10 dropped to 0.80 but nDCG@10 stayed the same. What happened?**
The 2 lost neighbours were ones float32 ranked highly but no human marked relevant.
The compressed answer differs from float32 without being worse for the user.
→ `design.md` §2.

**Q: Why does ternary store 2 bits but only carry about 1.6 bits of information?**
Two bits can hold 4 patterns; ternary uses 3 (`-1, 0, +1`), so one is wasted.
log2(3) ≈ 1.58. We accept it because the two-bit-plane layout (one mask for +1, one
for -1) is what makes popcount search fast. → `design.md` §3.

## Open

- Which exact Ollama model fits my machine's RAM for the Coder role? (Decide at
  Phase 2 start.)
- Should "bytes per vector" count the per-vector scale as 4 bytes (float32) or
  2 bytes (float16)? Decide and record before the first public numbers.
- Do we want a second embedding model in the first public release, or add it after?
- How many warm-up queries are enough for stable p50 on my laptop? (Measure.)
