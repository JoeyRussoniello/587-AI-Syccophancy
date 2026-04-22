from bert_score import BERTScorer


class BERTScoreScorer:
    """Pairwise semantic similarity using BERTScore F1.

    Intended use: compare LLM reasoning against crowd-sourced top_comment reasoning.
    References are the crowd comments; candidates are the LLM-generated sentences.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def score_pairs(self, references: list[str], candidates: list[str]):
        """Return BERTScore F1 for each (references[i], candidates[i]) pair."""
        scorer = BERTScorer(lang=self.lang, rescale_with_baseline=True)
        _, _, F1 = scorer.score(candidates, references)
        return F1
