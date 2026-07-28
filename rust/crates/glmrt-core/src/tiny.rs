pub fn deterministic_tiny_completion(prompt: &str, max_tokens: usize) -> String {
    let mut words = vec!["hello", "from", "glmrt", "tiny", "backend"];
    if prompt.to_ascii_lowercase().contains("count") {
        words = vec!["one", "two", "three"];
    }
    words.truncate(max_tokens.max(1));
    words.join(" ")
}
