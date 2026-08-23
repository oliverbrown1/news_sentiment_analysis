from __future__ import annotations


class ArticleExtractionError(RuntimeError):
    pass


class NewspaperArticleExtractor:
    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )

    def extract(self, url: str) -> str:
        from newspaper import Article, ArticleException, Config

        config = Config()
        config.browser_user_agent = self._user_agent
        article = Article(url, config=config)
        try:
            article.download()
            article.parse()
            article.nlp()
        except (ArticleException, LookupError) as exc:
            raise ArticleExtractionError(f"could not extract article: {url}") from exc

        summary = article.summary.strip()
        if not summary:
            raise ArticleExtractionError(f"article produced an empty summary: {url}")
        return summary
