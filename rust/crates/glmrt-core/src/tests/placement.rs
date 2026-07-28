use super::*;

#[test]
fn modulo_policy_matches_phase0_formula() {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        owner_for_expert(0, 0, &hosts, PlacementPolicy::Modulo).unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(0, 1, &hosts, PlacementPolicy::Modulo).unwrap(),
        "spark-1"
    );
    assert_eq!(
        owner_for_expert(1, 0, &hosts, PlacementPolicy::Modulo).unwrap(),
        "spark-0"
    );
}

#[test]
fn range_policy_splits_256_experts_four_ways() {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        owner_for_expert(7, 0, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(7, 63, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(7, 64, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-1"
    );
    assert_eq!(
        owner_for_expert(7, 255, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-3"
    );
}
