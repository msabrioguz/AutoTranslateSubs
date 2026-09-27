//! Token-aware batching for Ollama requests.
//!
//! Groups translation-ready items into request batches using an approximate
//! input-token budget instead of a fixed subtitle count alone. Estimates use
//! the same `chars / 4` formula the benchmark reports, so budget decisions
//! and reported estimates stay consistent.

/// Approximate token count for `text` (`chars / 4`, integer division).
///
/// A rough estimate, never an exact tokenizer count.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Groups `items` into batches under two limits:
///
/// * `max_input_tokens` — the accumulated estimated tokens of a batch may not
///   exceed this budget; an item that would push it over starts a new batch.
///   An item that exceeds the budget *by itself* is emitted as its own batch
///   and never dropped.
/// * `max_items` — hard ceiling on batch size (keeps LLM responses inside the
///   generation limit).
///
/// Returns the positions of `items` for every batch. Order is preserved and
/// every input position appears exactly once.
pub fn build_batches(
    items: &[(usize, &str)],
    max_input_tokens: usize,
    max_items: usize,
) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut used = 0usize;

    for (position, (_, text)) in items.iter().enumerate() {
        let estimated = estimate_tokens(text);
        let budget_full = used + estimated > max_input_tokens;
        let count_full = current.len() >= max_items;
        if !current.is_empty() && (budget_full || count_full) {
            batches.push(std::mem::take(&mut current));
            used = 0;
        }
        used += estimated;
        current.push(position);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items<'a>(texts: &[&'a str]) -> Vec<(usize, &'a str)> {
        texts.iter().enumerate().map(|(i, t)| (i, *t)).collect()
    }

    fn all_positions(batches: &[Vec<usize>]) -> Vec<usize> {
        batches.iter().flatten().copied().collect()
    }

    #[test]
    fn estimate_is_chars_divided_by_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens(&"x".repeat(9)), 2);
    }

    #[test]
    fn empty_input_yields_no_batches() {
        assert!(build_batches(&[], 3000, 25).is_empty());
    }

    #[test]
    fn single_item_forms_one_batch() {
        let items = items(&["only one line"]);
        assert_eq!(build_batches(&items, 3000, 25), vec![vec![0]]);
    }

    #[test]
    fn multiple_small_items_share_one_batch() {
        let texts: Vec<String> = (0..10).map(|i| format!("line {}", i)).collect();
        let refs: Vec<(usize, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.as_str()))
            .collect();
        let batches = build_batches(&refs, 3000, 25);
        assert_eq!(batches, vec![(0..10).collect::<Vec<_>>()]);
    }

    #[test]
    fn budget_boundary_is_inclusive() {
        // Each item estimates to 5 tokens (20 chars); two of them hit the
        // budget of 10 exactly and must stay together ("exceeded" splits).
        let a = "x".repeat(20);
        let b = "x".repeat(20);
        let items = items(&[&a, &b]);
        assert_eq!(build_batches(&items, 10, 25), vec![vec![0, 1]]);
    }

    #[test]
    fn splits_when_budget_would_be_exceeded() {
        let a = "x".repeat(20); // 5 tokens
        let b = "x".repeat(20); // 5 tokens
        let c = "x".repeat(20); // 5 tokens -> 10 + 5 > 10
        let items = items(&[&a, &b, &c]);
        assert_eq!(build_batches(&items, 10, 25), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn oversized_item_is_emitted_alone_and_kept() {
        let huge = "x".repeat(400); // 100 tokens > budget of 10
        let small = "tiny";
        let items = items(&[&huge, small, small]);
        let batches = build_batches(&items, 10, 25);
        assert_eq!(batches, vec![vec![0], vec![1, 2]]);
        assert_eq!(all_positions(&batches), vec![0, 1, 2]);
    }

    #[test]
    fn long_and_short_mixed_preserve_order_and_coverage() {
        let long = "x".repeat(200); // 50 tokens
        let short = "ab";
        let texts = [
            short.to_string(),
            long.clone(),
            short.to_string(),
            short.to_string(),
            long.clone(),
            short.to_string(),
        ];
        let refs: Vec<(usize, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.as_str()))
            .collect();
        let batches = build_batches(&refs, 60, 25);

        // Order preserved, every position exactly once
        assert_eq!(
            all_positions(&batches),
            (0..texts.len()).collect::<Vec<_>>()
        );
        // The 50-token long item does not fit next to a second long item
        // within the 60-token budget, so batches split around the long ones.
        assert!(batches.len() > 1);
    }

    #[test]
    fn max_items_ceiling_still_applies() {
        let texts: Vec<String> = (0..30).map(|i| format!("l{}", i)).collect();
        let refs: Vec<(usize, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.as_str()))
            .collect();
        let batches = build_batches(&refs, 1_000_000, 25);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 25);
        assert_eq!(batches[1].len(), 5);
    }
}
