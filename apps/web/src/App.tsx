import { useEffect, useMemo, useState } from "react";
import {
  addRating,
  getRecommendations,
  listItems,
  resetSession,
  startSession,
  VARIANTS,
  VARIANT_LABELS,
  type CatalogItem,
  type Rating,
  type Recommendation,
  type Variant,
} from "./api";

const emptyRecommendations = (): Record<Variant, Recommendation[]> => ({
  base: [],
  vector: [],
  matrix: [],
});

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [items, setItems] = useState<CatalogItem[]>([]);
  const [query, setQuery] = useState("");
  const [history, setHistory] = useState<Rating[]>([]);
  const [recommendations, setRecommendations] = useState(emptyRecommendations);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshRecommendations(id: string): Promise<void> {
    const arrays = await Promise.all(VARIANTS.map((v) => getRecommendations(id, v)));
    setRecommendations({ base: arrays[0], vector: arrays[1], matrix: arrays[2] });
  }

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const [newSession, catalog] = await Promise.all([startSession(), listItems()]);
        if (!active) return;
        setSessionId(newSession);
        setItems(catalog);
        const arrays = await Promise.all(VARIANTS.map((v) => getRecommendations(newSession, v)));
        if (active) setRecommendations({ base: arrays[0], vector: arrays[1], matrix: arrays[2] });
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      }
    }
    void initialize();
    return () => { active = false; };
  }, []);

  const alreadyRated = useMemo(() => new Set(history.map((h) => h.item_id)), [history]);
  const visible = items.filter(
    (item) =>
      item.title.toLowerCase().includes(query.toLowerCase()) ||
      item.categories.join(" ").toLowerCase().includes(query.toLowerCase()),
  );

  async function submitRating(itemId: string, stars: number) {
    if (!sessionId || busy || alreadyRated.has(itemId)) return;
    setBusy(true);
    setError(null);
    try {
      await addRating(sessionId, { item_id: itemId, rating: stars });
      setHistory((old) => [...old, { item_id: itemId, rating: stars }]);
      await refreshRecommendations(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (!sessionId || busy) return;
    setBusy(true);
    setError(null);
    try {
      await resetSession(sessionId);
      setHistory([]);
      await refreshRecommendations(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <p className="eyebrow">DEEP LEARNING SYSTEMS · LOCAL PREVIEW</p>
          <h1>NeedleTTT</h1>
          <p className="subtitle">How do recommendations respond after every rating?</p>
        </div>
        <button disabled={!sessionId || busy} onClick={() => void reset()} className="reset">
          Reset ratings
        </button>
      </header>

      <section className="warning" role="status">
        <strong>Untrained teaching model · fictional games.</strong> This starter exercises the
        interface only. Rankings are not trained recommendations or research results.
        Shared weights do not learn here; the vector and matrix fast states update from ratings.
      </section>

      {error && <section className="error" role="alert">{error}</section>}

      <section className="intro">
        <div><strong>1.</strong> Rate a few games below using 1–5 stars.</div>
        <div><strong>2.</strong> See the baseline, vector TTT and matrix TTT update side by side.</div>
        <div><strong>3.</strong> Replace the teaching adapter with the real Needle checkpoint later.</div>
      </section>

      <div className="layout">
        <section className="catalog">
          <div className="section-title">
            <h2>Discover games</h2>
            <span>{history.length} rated</span>
          </div>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search fictional games or genres"
            aria-label="Search games"
          />
          <div className="product-grid">
            {visible.map((item) => (
              <article className="product" key={item.item_id}>
                <div className="product-art" aria-label="No product image in the fictional sample">
                  <span aria-hidden>◇</span>
                </div>
                <div className="product-body">
                  <h3>{item.title}</h3>
                  <p>{item.categories.join(" · ")}</p>
                  <div className="stars" aria-label={"Rate " + item.title}>
                    {[1, 2, 3, 4, 5].map((rating) => (
                      <button
                        key={rating}
                        disabled={!sessionId || busy || alreadyRated.has(item.item_id)}
                        onClick={() => void submitRating(item.item_id, rating)}
                        aria-label={rating + " stars for " + item.title}
                      >
                        ★
                      </button>
                    ))}
                  </div>
                  {alreadyRated.has(item.item_id) && <small>Rated · hidden from next recommendations</small>}
                </div>
              </article>
            ))}
          </div>
          {visible.length === 0 && <p>No matching games.</p>}
        </section>

        <aside className="recommendation">
          <div className="section-title">
            <h2>Recommended next</h2>
            <span>Same candidate catalog</span>
          </div>
          <div className="result-grid">
            {VARIANTS.map((variant) => (
              <section className="result-column" key={variant}>
                <h3>{VARIANT_LABELS[variant]}</h3>
                <p className="variant-description">
                  {variant === "base"
                    ? "User representation changes with rating history."
                    : variant === "vector"
                      ? "A small user-specific vector takes gradient steps."
                      : "A user-specific matrix learns vector-valued responses."}
                </p>
                <ol>
                  {recommendations[variant].slice(0, 5).map((r) => (
                    <li key={r.item_id}>
                      <span>{r.title}</span>
                      <small>{r.score.toFixed(3)}</small>
                    </li>
                  ))}
                </ol>
                {recommendations[variant].length === 0 && <p>No eligible games left.</p>}
              </section>
            ))}
          </div>
          <div className="history">
            <strong>Observed feedback</strong>
            {history.length === 0 ? (
              <p>No ratings yet. The demo starts from identical empty histories.</p>
            ) : (
              <ol>
                {history.map((entry, index) => (
                  <li key={index}>
                    {items.find((x) => x.item_id === entry.item_id)?.title ?? entry.item_id}
                    <span>{"★".repeat(entry.rating)}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </aside>
      </div>
      <footer>
        Local app only · no public deployment · <a href="https://linear.app/deep-learning-system-final/project/needlettt-9724c36843f8/overview">Project on Linear</a>
      </footer>
    </div>
  );
}
