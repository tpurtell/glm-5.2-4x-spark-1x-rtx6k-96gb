use super::*;

#[test]
fn tiny_completion_is_deterministic() {
    assert_eq!(
        deterministic_tiny_completion("Count to three.", 16),
        "one two three"
    );
    assert_eq!(
        deterministic_tiny_completion("Say hello.", 3),
        "hello from glmrt"
    );
}
