# Interview walkthrough

## Two-minute version
1. **Problem:** deaf and mute users can follow captions, but replying quickly with personal facts is hard.
2. **Solution:** a LangGraph agent that captions speech, finds the answer in the user's notes and suggests a reply they tap to speak.
3. **Hard part:** a wrong medical fact is worse than no answer, so replies are checked by code and the system can abstain.
4. **Result:** semantic search took wrong facts from 19.5% to 0% on the test split, at the cost of abstaining on 12 of 29 answerable questions.
5. **Bug the demo caught:** a config path bug silently used the weak baseline, and "Do you have a pet?" returned a diabetes note. I fixed the path and made the demo show which search it is using.

## Questions to prepare
- **Why LangGraph instead of a chain?** The flow branches and loops: urgent speech skips the model, and failed verification retries once. It also keeps per-session history through a checkpointer.
- **Why would the cross-encoder not help recall?** The list was already short and the notes are small. It was built for web passages, not terse personal facts. Its score may still help with abstaining, but I haven't tested that alone.
- **Why did fixed chunks score higher?** Bigger chunks contain the answer more easily, so the metric flatters them. Precision of the spoken reply matters here.
- **What does the verifier miss?** It checks words and numbers, not meaning. "I am not allergic to penicillin" could pass against a penicillin note. An entailment check would be the next step.
- **Why did the harder intent set score 26.7%?** Rules only cover patterns I wrote. It's a held-out set that shows why the LLM is needed.
- **How would you fit this on real glasses?** Run Whisper on the phone or device, stream captions, and keep the notes on-device.

## Know these fundamentals
Embeddings and cosine similarity, how FAISS searches, tokens and context windows, recall@k and MRR, word error rate, dev and test splits, overfitting, prompt injection, and REST versus WebSockets.
